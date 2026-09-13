"""Prompt budget for small local models (Granite 3B default).

A 3B cannot follow a 800-token law-and-persona essay. Compact mode keeps
identity + one duty line. Set AMNI_PROMPT=full to restore the long prompts.
"""
import os

def use_compact_prompt()->bool:
    v=(os.environ.get('AMNI_PROMPT') or os.environ.get('AMNI_COMPACT_PROMPT') or 'compact').strip().lower()
    return v not in ('0','false','no','full','off')

def history_turns(default_full:int=12)->int:
    env=os.environ.get('AMNI_HISTORY_TURNS')
    if env and env.isdigit():return max(1,int(env))
    return 6 if use_compact_prompt() else default_full

_COT={
    'math':'Math problem — name the formula, plug the numbers, give the answer with units. No step labels.',
    'code':'Code task — write the language they asked for in one fenced block. No prelude.',
    'debug':'Debugging — one likely cause, one check to run. No five-step ritual.',
    'design':'System design — three named parts and the main trade-off. Short.',
    'reasoning':'Reasoning question — one claim, one reason, the answer. No RESTATE labels.',
    'explain':'Explain-it — two short paragraphs. Lead with the answer.',
    'generic':'Answer the question. If a fact below is on file, use it.',
}

def compact_cot(kind:str)->str:
    return _COT.get(kind or 'generic',_COT['generic'])
