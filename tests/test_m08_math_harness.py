import datetime as dt,sys
from fractions import Fraction as Fr
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from amni.math import answer,solve_math,solve,exec_calc,apply_protocol,classify
from amni.math.relate import compile_problem,solve_lin,V,C
import m08cto_eval
T=dt.date(2026,9,25)
jail=pytest.mark.skipif(not (sys.platform.startswith('linux') and Path('/usr/bin/bwrap').exists()),reason='sympy runs only inside the bwrap jail')
@pytest.mark.parametrize('q,v',[("What is 15% of 80?",Fr(12)),("12 times 7 plus 3",Fr(87)),("what is 1/3 + 1/4",Fr(7,12)),("0.1+0.2",Fr(3,10)),("square root of 144",Fr(12)),("5!",Fr(120)),("what is twenty five times four",Fr(100)),("What percent of 80 is 12?",Fr(15)),("12 is what percent of 80",Fr(15)),("30 is 20% of what?",Fr(150)),("increase 80 by 15%",Fr(92)),("20% off $50",Fr(40)),("percent change from 80 to 100",Fr(25)),("how many days between March 3, 2026 and April 17, 2026",Fr(45)),("what day of the week is 2026-12-25",Fr(4)),("how many days until christmas",Fr(91)),("solve 3x+4=19",Fr(5)),("average of 3, 4 and 5",Fr(4)),("gcd of 12 and 18",Fr(6)),("sum of 1 to 100",Fr(5050))])
def test_exact_values(q,v):
 r=solve_math(q,T);assert r is not None and r['value']==v,(q,r)
@pytest.mark.parametrize('q,v',[("convert 10 km to miles",10000/1609.344),("100 fahrenheit in celsius",(100-32)*5/9),("how many feet in 3 miles",15840.0),("how many seconds in a day",86400.0)])
def test_units(q,v):
 r=solve_math(q,T);assert r is not None and abs(float(r['value'])-v)<=1e-9*max(1,v),(q,r)
def test_reply_shows_working():
 assert answer("What is 15% of 80?",T)[0]=="15% of 80 is 12 (0.15 × 80)"
 assert "(12 ÷ 80 × 100)" in answer("12 is what percent of 80",T)[0] and "=" in answer("12*7+3",T)[0]
@pytest.mark.parametrize('q,v',[("Paddington has 40 more goats than Washington. If Washington has 140 goats, how many goats do they have in total?",Fr(320)),("Tom has 5 apples. He buys 3 more apples. How many apples does Tom have now?",Fr(8)),("Maria has $50. She spends $12 on lunch and $8 on a book. How much money does she have left?",Fr(30)),("Jake has 3 more apples than Steven. Steven has 5 apples. How many apples does Jake have?",Fr(8)),("Emily has 7 times as many oranges as Sandra. Sandra has 3 times as many oranges as Betty. If Betty has 12 oranges, how many oranges does Emily have?",Fr(252)),("Mary has 400 sheep on her farm. She gave a quarter of her sheep to her sister, and half of the remaining sheep to her brother. How many sheep remain with Mary?",Fr(150))])
def test_word_problems_verified(q,v):
 r=solve_math(q,T);assert r is not None and r['intent']=='word' and r['value']==v and 'Working:' in r['reply'],(q,r)
 c=compile_problem(q);assert c['verified'] and c['all_used'] and not c['unparsed']
def test_word_declines_unverifiable():
 assert solve_math("A train leaves at 3 pm going 60 mph while another leaves later; when do they meet if one is faster?",T) is None
@pytest.mark.parametrize('q',["What is the capital of France?","How many moons does Jupiter have?","Who was the 16th president?","What year did World War 2 end?","What is 1984 about?","What happened on 9/11?","2026-09-25","hello how are you","How many people live in Tokyo?","Write me a poem about 2 cats."])
def test_router_leaves_non_math(q):
 assert answer(q,T) is None,q
def test_classify_intents():
 assert 'percent' in classify("What is 15% of 80?")['intents'] and 'date' in classify("how many days until christmas")['intents'] and classify("What is the capital of France?")['intents']==[]
def test_solve_lin_exact():
 s=solve_lin([V('a')-2*V('b'),V('b')-C(7)]);assert s=={'a':Fr(14),'b':Fr(7)}
 assert solve_lin([V('a')-C(1),V('a')-C(2)]) is None
def test_harness_api():
 a,tr=solve("Paddington has 40 more goats than Washington. If Washington has 140 goats, how many goats do they have in total?");assert a=='320' and tr[0]['conf']>=0.6 and not tr[0]['hedged']
 assert solve("hello") is None and solve("") is None
def test_protocol_and_calc():
 assert exec_calc("12*7+3")=="87" and exec_calc("2^10+3")=="1027" and exec_calc("1/3")=="0.333333333333"
 out,tr=apply_protocol("ans ⟦calc: 3+4⟧ done");assert out=="ans 7 done" and tr[0]['value']=='7'
@jail
def test_algebra_via_jail():
 assert answer("solve x^2-5x+6=0",T)[0].startswith("x = 2 or x = 3") and solve_math("solve x+y=10, x-y=2",T)['reply'].startswith("x = 6, y = 4")
 assert "(x - 1)*(x + 1)" in answer("factor x^2-1",T)[0] and "3*x^2" in answer("derivative of x^3",T)[0]
def test_suite_generator_independent_gold():
 S=m08cto_eval.gen_suite();assert len(S)==500 and {x['kind'] for x in S}=={'arith','percent','units','date','algebra'}
 assert m08cto_eval.match(Fr(1,3),Fr(1,3)) and not m08cto_eval.match(Fr(1,3),Fr(1,4)) and not m08cto_eval.match(None,Fr(1))
def test_test_split_needs_one_look_flag():
 with pytest.raises(SystemExit,match='one look'):m08cto_eval.main(['test'])
