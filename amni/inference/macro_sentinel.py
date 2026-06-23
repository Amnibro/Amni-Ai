"""MacroSentinel — the macro method INVERTED for security/debugging (Anthony's "mini mini adam detects threats"). Same coverage() primitive as the codegen accelerator, pointed at low-level streams (syscall traces, instruction blocks, packet flows, kernel logs). Two modes, both deterministic + microsecond-fast: (1) ANOMALY = coverage against a known-good macro allow-list; uncovered spans = where the threat/bug IS (localized). (2) SIGNATURE = known-bad macro match (shellcode/W^X/exploit sequences). Structural skeletons (normalize_line) defeat polymorphism. The tiny model only adjudicates the small UNCOVERED residual, never the firehose -> real-time. Compounds: learn_normal() shrinks false positives, add_signature() grows coverage. Built on MacroCodeEngine."""
from amni.inference.macro_code_engine import MacroCodeEngine
class MacroSentinel(MacroCodeEngine):
    def __init__(s,max_block=3,min_freq=2,anomaly_thresh=75.0):
        super().__init__(None,max_block=max_block,min_freq=min_freq,min_len=3);s.bad={};s.anomaly_thresh=anomaly_thresh
    def learn_normal(s,traces):return s.mine([t.strip() for t in traces])
    def add_signature(s,pattern,name):s.bad[pattern]=name;return len(s.bad)
    def scan(s,stream):
        lines=[L.strip() for L in stream.split('\n')];i=0;covered=0;anomalies=[]
        total=sum(1 for L in lines if L)
        while i<len(lines):
            if not lines[i]:i+=1;continue
            m=0
            for L in range(min(s.max_block,len(lines)-i),0,-1):
                if '\n'.join(lines[i:i+L]) in s.blocks:m=L;break
            if m:covered+=m;i+=m
            else:anomalies.append((i,lines[i]));i+=1
        sigs=[(j,s.bad[p],ln) for j,ln in enumerate(lines) if ln for p in s.bad if p in ln]
        cov=round(100*covered/max(total,1),1)
        return {'coverage':cov,'total':total,'anomalies':anomalies,'signatures':sigs,'residual':len(anomalies),'threat':bool(sigs) or cov<s.anomaly_thresh}
