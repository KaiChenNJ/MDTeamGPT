from typing import TypedDict, List, Annotated, Any
import operator
from langgraph.graph import StateGraph, END
from knowledge_base import kb_system


class MDTState(TypedDict):
    case_info: str
    image_base64: str
    ground_truth: str

    selected_roles: List[str]
    triage_reason: str

    current_round: int
    max_rounds: int

    context_bullets: Annotated[List[str], operator.add]
    final_answer: str
    is_converged: bool

    kb_context_text: str
    kb_context_docs: Any


def create_workflow(agents_instance):
    def node_triage(state: MDTState):
        kb_system.init_embeddings(
            api_key=agents_instance.llm.openai_api_key,
            base_url=agents_instance.llm.openai_api_base
        )

        retrieval_result = kb_system.retrieve_context_details(state["case_info"])
        triage_result = agents_instance.primary_care_doctor(state["case_info"])

        return {
            "selected_roles": triage_result["selected_roles"],
            "triage_reason": triage_result["reasoning"],
            "current_round": 1,
            "kb_context_text": retrieval_result["text"],
            "kb_context_docs": retrieval_result["docs"],
            "context_bullets": []
        }

    def node_consultation_and_synthesis(state: MDTState):
        roles = state["selected_roles"]
        rnd = state["current_round"]
        bullets = state["context_bullets"]

        #  Logic Check: Residual Context
        # 1. This is calculated BEFORE the agent loop.
        # 2. It only contains info from PREVIOUS rounds (bullets).
        # 3. Therefore, agents in this round CANNOT see each other's current output.
        residual_context = ""
        if rnd == 1:
            residual_context = f"PRIOR KNOWLEDGE FROM DB:\n{state['kb_context_text']}"
        else:
            recent_bullets = bullets[-2:]
            for i, b in enumerate(recent_bullets):
                bullet_rnd = rnd - len(recent_bullets) + i
                residual_context += f"--- Round {bullet_rnd} Summary ---\n{b}\n"

        dialogues = []
        for role in roles:
            img = state["image_base64"] if rnd == 1 else None

            # Logic Check: Independence & Blindness
            # 1. 'residual_context' is static for all agents in this loop.
            # 2. 'ground_truth' is NOT passed to the agent.
            res = agents_instance.specialist_consult(
                role, state["case_info"], residual_context, img, rnd
            )
            dialogues.append(f"**{role}**: {res}")

        # Lead Physician synthesizes the accumulated dialogues
        summary_json = agents_instance.lead_physician_synthesis(dialogues, rnd)

        return {
            "context_bullets": [summary_json],
            "current_round": rnd
        }

    def node_safety_check(state: MDTState):
        last_bullet = state["context_bullets"][-1]
        rnd = state["current_round"]

        # Safety Reviewer checks convergence based on the summary
        review = agents_instance.safety_reviewer(last_bullet, rnd)

        is_converged = "STATUS: CONVERGED" in review
        final_ans = ""

        if "FINAL_ANSWER:" in review:
            parts = review.split("FINAL_ANSWER:")
            final_ans = parts[1].strip() if len(parts) > 1 else review

        if rnd >= state["max_rounds"]:
            is_converged = True
            if not final_ans:
                final_ans = "Max rounds reached. Proceeding with latest hypothesis."

        return {
            "is_converged": is_converged,
            "final_answer": final_ans,
            "current_round": rnd + 1
        }

    def router(state: MDTState):
        if state["is_converged"]:
            return "end"
        return "continue"

    workflow = StateGraph(MDTState)

    workflow.add_node("triage", node_triage)
    workflow.add_node("consultation_layer", node_consultation_and_synthesis)
    workflow.add_node("safety_layer", node_safety_check)

    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "consultation_layer")
    workflow.add_edge("consultation_layer", "safety_layer")

    workflow.add_conditional_edges("safety_layer", router, {"continue": "consultation_layer", "end": END})

    return workflow.compile()