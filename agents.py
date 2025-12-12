from typing import List, Dict, Any, Callable
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tools import MedicalTools

SPECIALIST_POOL = [
    "General Internal Medicine Doctor",
    "General Surgeon",
    "Pediatrician",
    "Obstetrician and Gynecologist",
    "Radiologist",
    "Neurologist",
    "Pathologist",
    "Pharmacist"
]


class MDTAgents:
    def __init__(self, api_key, base_url, text_model, vl_model, enable_tools=True):
        self.llm = ChatOpenAI(
            model=text_model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.7,
            streaming=True
        )
        self.critic_llm = ChatOpenAI(
            model=text_model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            streaming=False
        )
        self.vl_llm = ChatOpenAI(
            model=vl_model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.1,
            max_tokens=2048,
            streaming=True
        )

        self.tools = MedicalTools(enable=enable_tools)
        # Callbacks
        self.stream_callback = None
        self.tool_callback = None

    def set_stream_callback(self, callback: Callable[[str, str], None]):
        self.stream_callback = callback

    def set_tool_callback(self, callback: Callable[[str, str, str], None]):
        self.tool_callback = callback

    # 1. Primary Care (Triage)
    def primary_care_doctor(self, case_info: str) -> Dict[str, Any]:
        prompt = ChatPromptTemplate.from_template(
            """You are a Primary Care Doctor at the Triage Desk.
            Analyze the patient case and select the most appropriate specialists.

            Available Specialists:
            {pool}

            Patient Case: {case}

            TASK:
            1. Explain your reasoning.
            2. Select AT LEAST 3 specialists.

            OUTPUT JSON FORMAT:
            {{
                "reasoning": "...",
                "selected_roles": ["Role A", "Role B", "Role C"]
            }}
            """
        )
        chain = prompt | self.llm
        result = chain.invoke({"pool": ", ".join(SPECIALIST_POOL), "case": case_info})

        content = result.content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.endswith("```"): content = content[:-3]

        try:
            data = json.loads(content)
            selected = [s for s in data.get("selected_roles", []) if s in SPECIALIST_POOL]
            remaining = [s for s in SPECIALIST_POOL if s not in selected]
            while len(selected) < 3 and remaining:
                selected.append(remaining.pop(0))
            data["selected_roles"] = selected
            return data
        except:
            return {
                "reasoning": "Fallback selection.",
                "selected_roles": ["General Internal Medicine Doctor", "General Surgeon", "Radiologist"]
            }

    #2. Specialists (Consultation)
    def specialist_consult(self, role: str, case_info: str, residual_context: str,
                           image_data=None, round_num=1):

        #Tool Usage Logic
        tool_context = ""
        if self.tools.enable:
            try:
                kw_prompt = ChatPromptTemplate.from_template(
                    "Extract 1 specific medical query string for {role} to research regarding: {case}. Return ONLY the query.")
                kw_chain = kw_prompt | self.critic_llm
                kw = kw_chain.invoke({"case": case_info[:300], "role": role}).content

                if kw and "no query" not in kw.lower():
                    tool_res = self.tools.run_tools(kw)
                    if tool_res:
                        if self.tool_callback:
                            self.tool_callback(role, kw, tool_res)
                        tool_context = f"\n[External Tools Data]:\n{tool_res}\n"
            except Exception as e:
                print(f"Tool error: {e}")

        # Strict Reasoning Structure
        structure_instruction = """
        IMPORTANT INSTRUCTIONS:
        1. **Independence**: You are providing your opinion INDEPENDENTLY. You cannot see the opinions of other specialists in this current round. You can only see the summary of previous rounds (if any).
        2. **Blindness**: You do NOT have access to the ground truth or final correct diagnosis. Rely only on the case description and your knowledge.
        3. **Structure**: You must structure your response in exactly three sections:

           - **1. Context Summary**: 
             (If Round 1: Summarize "Prior Knowledge". If Round > 1: Summarize "Residual Context" from previous rounds.)

           - **2. Clinical Reasoning**: 
             (Analyze the case. If tool data exists, use it. If image exists, describe findings. Explain step-by-step.)

           - **3. Conclusion**: 
             (State your clear medical opinion or diagnosis.)
        """

        system_prompt = f"You are a {role}. Provide expert medical opinion.\n{structure_instruction}"

        user_text = f"Patient Case: {case_info}\n{tool_context}\n"

        if round_num == 1:
            user_text += "\n[Status]: Round 1. Analyze independently."
            user_text += f"\n*** PRIOR KNOWLEDGE / CONTEXT ***\n{residual_context}\n"
            if image_data:
                user_text += " [Image Provided]. Describe findings and integrate with diagnosis."
            else:
                user_text += " No image provided."
        else:
            user_text += f"\n[Status]: Round {round_num}.\n"
            user_text += f"*** RESIDUAL CONTEXT (Previous Rounds) ***\n{residual_context}\n"
            user_text += "Review the summaries of previous rounds. Support, refute, or synthesize based on that history."

        messages = [SystemMessage(content=system_prompt)]

        target_llm = self.llm
        if round_num == 1 and image_data:
            target_llm = self.vl_llm
            img_url = f"data:image/jpeg;base64,{image_data}"
            content_payload = [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": img_url}}
            ]
            messages.append(HumanMessage(content=content_payload))
        else:
            messages.append(HumanMessage(content=user_text))

        try:
            full_res = ""
            for chunk in target_llm.stream(messages):
                token = chunk.content
                full_res += token
                if self.stream_callback: self.stream_callback(role, token)
            return full_res
        except Exception as e:
            return f"Error: {e}"

    #3. Lead Physician
    def lead_physician_synthesis(self, round_dialogues: List[str], round_num: int):
        # Lead Physician DOES see all dialogues from the current round (to synthesize them),
        # but DOES NOT see Ground Truth.
        prompt = ChatPromptTemplate.from_template(
            """You are the Lead Physician.
            Synthesize the specialists' discussions from Round {rnd} into a concise structured summary.

            Specialists' Output (Current Round):
            {dialogues}

            TASK:
            Create a JSON object containing EXACTLY these 6 fields:

            1. "Consistency": (Aggregates the parts of individual statements that are consistent across multiple agent statements).
            2. "Conflict": (Identifies conflicting points between statements; empty if none).
            3. "Independence": (Extracts unique viewpoints of each agent not mentioned by others).
            4. "Integration": (Synthesizes all statements into a cohesive summary).
            5. "Tools_Usage": (Summarize specific tools/searches used in this round).
            6. "Long_Term_Experience": (Extract and summarize any prior experience/knowledge referenced from the database).

            Return ONLY valid JSON.
            """
        )
        chain = prompt | self.llm
        res = chain.invoke({
            "rnd": round_num,
            "dialogues": "\n\n".join(round_dialogues)
        })

        content = res.content.strip()
        if content.startswith("```json"): content = content[7:]
        if content.endswith("```"): content = content[:-3]
        return content.strip()

    #4. Safety Reviewer
    def safety_reviewer(self, current_bullet: str, round_num: int):
        prompt = ChatPromptTemplate.from_template(
            """You are the Safety and Ethics Reviewer.
            Review the current round's synthesis.

            Current Context:
            {bullet}

            TASK:
            Determine if the medical diagnosis has converged to a solid, safe conclusion without major conflicts.

            OUTPUT FORMAT (Strict):
            STATUS: [CONVERGED / DIVERGED]
            REASON: [Short explanation]
            FINAL_ANSWER: [The final diagnosis/answer if converged, else "Continuing discussion"]
            """
        )
        chain = prompt | self.critic_llm
        res = chain.invoke({"bullet": current_bullet})
        return res.content

    # 5. CoT Reviewer
    def cot_reviewer(self, case_info, final_answer, ground_truth):
        # Only this agent sees the Ground Truth
        prompt = ChatPromptTemplate.from_template(
            """You are the 'Chain-of-Thought Reviewer'.

            CASE: {case}
            MODEL ANSWER: {answer}
            GROUND TRUTH: {truth}

            TASK:
            Step 1: Determine correctness (letters match for Choice, semantic match for Open).

            Step 2: Generate specific fields based on correctness.

            IF CORRECT:
               - "is_correct": true
               - "summary_s4": A concise summary of the final reasoning (S4_final).

            IF INCORRECT:
               - "is_correct": false
               - "initial_hypothesis": What was the likely first thought?
               - "analysis_process": Step-by-step breakdown of the failure.
               - "final_conclusion": The wrong conclusion reached.
               - "error_reflection": Why it was wrong and how to avoid it.

            OUTPUT JSON ONLY.
            """
        )
        chain = prompt | self.critic_llm
        try:
            res = chain.invoke({
                "case": case_info[:500],
                "answer": final_answer,
                "truth": ground_truth
            })
            content = res.content.strip()
            if content.startswith("```json"): content = content[7:]
            if content.endswith("```"): content = content[:-3]
            return json.loads(content)
        except:
            return {"is_correct": False, "analysis_text": "Parse Error"}