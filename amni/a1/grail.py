import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time

from amni.a1.triumvirate import Triumvirate, CouncilVerdict

from amni.a1.dual_mind import DualMind
from amni.a1.lawkeeper import LawKeeper
from amni.a1.delta_writer import DeltaWriter

# =============================================================================
# Phase 1: Damping Loop (Pydantic Models & Middleware)
# =============================================================================

class ProposedAction(BaseModel):
    intent: str
    target_file: Optional[str] = None
    code_snippet: str = ""
    domain_tags: List[str] = Field(default_factory=list)

class DampingSignal(BaseModel):
    rejected: bool
    reason: str
    structural_suggestion: Optional[str] = None

class AuditResult(BaseModel):
    passed: bool
    syntax_errors: List[str] = Field(default_factory=list)
    law_violations: List[str] = Field(default_factory=list)

import re

def extract_json(text: str) -> dict:
    try:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
        else:
            parsed = json.loads(text)
            
        # Unwrap if the LLM nested it under the class name
        if "ProposedAction" in parsed and isinstance(parsed["ProposedAction"], dict):
            return parsed["ProposedAction"]
        if "DampingSignal" in parsed and isinstance(parsed["DampingSignal"], dict):
            return parsed["DampingSignal"]
            
        return parsed
    except Exception:
        return {}

class DampingLoop:
    """
    Middleware that bounces ProposedAction from Left Brain to Right Brain & Cerebellum.
    Halts and returns or escalates if consensus fails.
    """
    def __init__(self, lawkeeper, delta_writer):
        self.lawkeeper = lawkeeper
        self.delta_writer = delta_writer
        self.max_iterations = 3

    async def execute(self, task: str, left_brain: Any, right_brain: Any) -> Optional[ProposedAction]:
        iteration = 0
        current_feedback = "Initial Task: " + task

        # System prompts to enforce JSON schema
        left_sys = (
            "You are the Left Brain Planner/Executor. Propose an action matching the following JSON schema strictly:\n"
            '{\n  "intent": "string (your goal)",\n  "target_file": "string or null",\n  "code_snippet": "string (the code)",\n  "domain_tags": ["list of strings"]\n}\n'
            "Output ONLY valid JSON."
        )
        right_sys = (
            "You are the Right Brain Challenger. Review the Left Brain's ProposedAction. Output matching the following JSON schema strictly:\n"
            '{\n  "rejected": true/false,\n  "reason": "string (why it passed or failed)",\n  "structural_suggestion": "string or null"\n}\n'
            "Output ONLY valid JSON."
        )

        while iteration < self.max_iterations:
            # 1. Left Brain Proposes
            messages_left = [
                {"role": "system", "content": left_sys},
                {"role": "user", "content": current_feedback}
            ]
            
            # Assuming left_brain and right_brain are DualMind._MiniModel instances with a .gen() method
            # In a real environment, left_brain.gen() is blocking, so we use to_thread
            if hasattr(left_brain, "gen"):
                print(f"[DampingLoop] Left Brain thinking (Iteration {iteration})...")
                proposal_resp = await asyncio.to_thread(left_brain.gen, messages_left, max_new=1024, temp=0.5)
                proposal_text = proposal_resp.get("raw", "")
                print(f"[DampingLoop] Left Brain output: {proposal_text[:100]}...")
            else:
                # Mock fallback
                proposal_text = '{"intent": "mock", "code_snippet": "print(1)"}'

            parsed_proposal = extract_json(proposal_text)
            try:
                proposal = ProposedAction(**parsed_proposal)
            except Exception as e:
                iteration += 1
                current_feedback = f"Iteration {iteration} failed.\nLeft Brain JSON Parsing Error: {e}\nRaw output: {proposal_text}"
                print(f"[DampingLoop] Left Brain failed validation: {e}")
                continue
            
            # 2. Right Brain Challenges (Structural Analogy)
            messages_right = [
                {"role": "system", "content": right_sys},
                {"role": "user", "content": f"Task: {task}\nLeft Brain Proposal:\n{proposal_text}"}
            ]
            if hasattr(right_brain, "gen"):
                print(f"[DampingLoop] Right Brain thinking (Iteration {iteration})...")
                damping_resp = await asyncio.to_thread(right_brain.gen, messages_right, max_new=1024, temp=0.3)
                damping_text = damping_resp.get("raw", "")
                print(f"[DampingLoop] Right Brain output: {damping_text[:100]}...")
            else:
                damping_text = '{"rejected": false, "reason": "Looks good."}'

            parsed_damping = extract_json(damping_text)
            try:
                damping = DampingSignal(**parsed_damping)
            except Exception as e:
                damping = DampingSignal(rejected=True, reason=f"Right Brain failed to output valid JSON: {e}")

            # 3. Cerebellum Audits (LawKeeper & Sandbox)
            if proposal.target_file:
                can_write, reason = self.lawkeeper.can_write(proposal.target_file)
                if not can_write:
                    audit = AuditResult(passed=False, law_violations=[reason])
                else:
                    audit = AuditResult(passed=True)
            else:
                audit = AuditResult(passed=True)
            
            # 4. Resolve Ping-Pong
            if not damping.rejected and audit.passed:
                return proposal
            
            # Compile feedback for next Left Brain iteration
            iteration += 1
            current_feedback = f"Iteration {iteration} failed.\nRight Brain: {damping.reason}\nCerebellum: {audit.law_violations}"
            
        return None # Local impasse reached

# =============================================================================
# Phase 2: Domain Acquisition Engine (Machine 1)
# =============================================================================

class DomainAcquisitionEngine:
    """
    Machine 1: I/O Bound. Formulates domain questions and seals the Triumvirate consensus.
    """
    def __init__(self, triumvirate: Triumvirate, delta_writer: DeltaWriter):
        self.triumvirate = triumvirate
        self.delta_writer = delta_writer

    async def acquire_domain(self, identified_gap: str):
        # Convert specific gap into maximal domain query
        prompt = (
            f"We lack fundamental knowledge regarding: {identified_gap}. "
            "Provide a complete mental model, functional abstractions, and professional methodologies "
            "for this entire domain, formatted for permanent vector storage."
        )
        failure_state = {"gap": identified_gap, "local_attempts_failed": True}
        
        # Escalate to Oracle Council
        verdict = await self.triumvirate.escalate(failure_state, prompt)
        
        if verdict.hallucination_filtered:
            # Oracle Distillation Pipeline (Sealing)
            self._seal_knowledge(identified_gap, verdict.consensus_text)

    def _seal_knowledge(self, domain: str, knowledge: str):
        # Mocking the DeltaWriter extraction
        if hasattr(self.delta_writer, 'add_creativity'):
            # This would parse the text and extract specific patterns
            # Here we just dump a chunk
            self.delta_writer.add_creativity(
                pattern=knowledge[:200], 
                context=f"domain_acquisition:{domain}", 
                quality=0.9
            )
            print(f"[Machine 1] Sealed domain {domain} into Atlas.")

# =============================================================================
# Phase 4: Boot Sequence
# =============================================================================

class BootSequence:
    def __init__(self, triumvirate: Triumvirate, delta_writer: DeltaWriter):
        self.triumvirate = triumvirate
        self.delta_writer = delta_writer

    async def run(self):
        print("[BootSequence] Firing Primordial Prompt 1 (Plumbing)...")
        p1 = "Design communication schemas for ProposedAction, DampingSignal, and AuditResult. Respond in JSON."     
        v1 = await self.triumvirate.escalate({"boot": "phase1"}, p1)
        # Avoid truncating the pattern string, allowing the full json to be sealed. Wait 5s to avoid hitting API rate limits with Gemini.
        self.delta_writer.add_creativity(pattern=v1.consensus_text, context="system:plumbing", quality=1.0)    
        await asyncio.sleep(5)

        print("[BootSequence] Firing Primordial Prompt 2 (Engine)...")
        p2 = "Design Structural Fingerprinting (5-axis) and Mutation Operators (Graft, Pipeline, Hybridize, Invert)."
        v2 = await self.triumvirate.escalate({"boot": "phase2"}, p2)
        self.delta_writer.add_creativity(pattern=v2.consensus_text, context="system:mutation_engine", quality=1.0)

        print("[BootSequence] System Bootstrapped.")
async def main():
    print("Amni-Grail Scaffold Initiated.")

if __name__ == "__main__":
    asyncio.run(main())
