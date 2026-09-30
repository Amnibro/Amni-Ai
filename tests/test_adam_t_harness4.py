import os,sys,shutil
sys.path[:0]=[os.path.dirname(os.path.dirname(os.path.abspath(__file__))),os.path.dirname(os.path.abspath(__file__))]
from fractions import Fraction as Fr
import pytest
import amni.compute
from amni.math import relate as RL
from amni.math.answer import solve_math
from amni.serve import adam_t_harness as H,adam_t_code as K,gguf_runtime as G
from test_adam_t_route import fake,rules,prompts
from test_adam_t_harness import FC,on
JAIL=pytest.mark.skipif(not (sys.platform.startswith('linux') and shutil.which('bwrap')),reason='code checks run only inside the bwrap sandbox')
@pytest.mark.parametrize('q,v',[("Kim had 14 stickers, lost 6 and then found 3 more. How many stickers does Kim have now?",11),("A shelf holds 5 rows of books with 9 books in each row. How many books are on the shelf?",45),("There are 30 grapes in a bowl. If 5 kids share them equally, how many grapes does each kid get?",6),("Tickets cost $7 each. How much do 4 tickets cost?",28),("A tank had 50 fish. 12 were sold and 8 new fish arrived. How many fish are in the tank?",46),("Rita is 9 years old. Her cousin is 4 years older. How old is her cousin?",13),("A jar has 40 marbles. Half of them are blue. How many marbles are not blue?",20),("Ian walks 3 miles each day. How many miles does he walk in a week?",21)])
def test_word_problem_fallback_solves_simple_frames(q,v):
    r=solve_math(q);assert r is not None and r['intent']=='word' and r['value']==Fr(v) and 'Working:' in r['reply'],(q,r)
@pytest.mark.parametrize('q',["I have 2 kids and 3 dogs. How many dogs do I have?","The band had 4 members and released 13 albums. How many albums did they release?","In 1990 there were 12 teams. How many teams are there now?","I have 5 quarters and 3 dimes. How many cents do I have?","Sara has 5 apples and Tom has 3 apples. How many apples does Sara have in total with Ben's 4?","It takes 2 hours to paint 1 room. How much does she earn per hour?"])
def test_word_problem_fallback_declines_unclear_frames(q):
    r=RL._seq(q);assert r is None,(q,r)
def test_word_problem_routes_to_solver_before_memory(on):
    cl=FC('model text');r=H.run("If I have 6 pears and eat two, then buy 4 more, how many do I have?",[],'w','w',cl,160)
    assert r['slots'][0]['kind']=='calc' and r['slots'][0]['ok'] and '8' in r['text'] and "haven't told" not in r['text'] and r['slots'][0]['mroute']=='calc' and r['slots'][0]['blend'] is False and cl.P[0][1][0].startswith('Calculator: ')
    assert H._wp("A classroom has 20 kids. Half of them are boys. How many girls are there?") and not H._wp("How many kids do I have?") and not H._wp("What is the capital of France?")
    cl.t="You would have 8 pears.";r=H.run("If I have 6 pears and eat two, then buy 4 more, how many do I have?",[],'w','w',cl,160);assert r['text']=="You would have 8 pears." and r['slots'][0]['blend'] is True and r['n']['hbp']==1
def test_facts_in_the_same_message_are_stored_first(on):
    cl=FC('model text');r=H.run("I'm Lena and I collect stamps. What do I collect?",[],'f1','f1',cl,160);assert 'stamps' in r['text'].lower() and r['slots'][0]['kind']=='mem' and r['slots'][0]['ok']
    r=H.run("What's my name again?",[],'f1','f1',cl,160);assert r and 'Lena' in r['text']
    assert H._stm("My dog is Rex, what's my dog's name?")==["My dog is Rex"] and H._stm("If I have 3 pears and eat one, how many are left?")==[] and H._stm("Write me a poem. What rhymes with cat?")==[]
    r=H.run("My dog is named Max. What is my cat called?",[],'f2','f2',cl,160);assert r['text']=="You haven't told me that yet."
def test_guard_rejects_templates_and_boilerplate():
    assert H.guard("I had a long week.","Tell them what your child looks like, what they were wearing, and where you last saw them.")=='' and H.guard("hi there","I'm sorry, but I am not able to respond to that.")=='' and H.guard("hey","I am a language model, so I can help.")==''
    assert H.guard("I had a long week.","That sounds tiring. Want to talk about it?")=="That sounds tiring. Want to talk about it?" and H.guard("how do I commute?","You told me you just started.")=="You told me you just started."
    assert H.safe("I had a long week.","Press hard on the cut with a clean cloth and keep pressing.")=="Thanks for sharing that. Tell me more?" and H.safe("x","")==''
def test_guard_on_model_path_keeps_emergency_first(fake,rules,monkeypatch):
    monkeypatch.setenv('AMNI_ADAM_T_HARNESS','1');monkeypatch.setenv('FAKE_MODE','say911');r=G.chat("My cat keeps sleeping on my keyboard.",user='g4');assert r['tier']=='tier_adam_t' and r['answer']=="Thanks for sharing that. Tell me more?"
    monkeypatch.setenv('AMNI_ADAM_T_HARNESS','0');r=G.chat("My cat keeps sleeping on my keyboard.",user='g4');assert r['answer'].strip()=="Kittens are fun."
    r=G.chat("My mom and my dad are laying on the floor and do not answer me. I'm scared.",user='g4');assert r['tier']=='tier_emergency' and '911' in r['answer']
def test_blend_gate_keeps_material_and_must_terms():
    M=H.mat('calc',['Calculator: The answer is 12. Working: 3 x 4 = 12.'],'The answer is 12. Working: 3 x 4 = 12.',['calculator: word'],0.8,['12'],echo='');q='Three bags hold four apples each, how many apples?'
    assert H.blend(FC('You get 12 apples. Anything else I can help with?'),q,M,200)==('You get 12 apples.',True) and H.blend(FC('You get 13 apples.'),q,M,200)==(M['extract'],False) and H.blend(FC('x'),q,{**M,'lines':[]},200)==(M['extract'],None)
    N=H.mat('mem',['You told me: My sister lives in Oslo.'],'Your sister lives in Oslo.',['user memory'],0.9,['Oslo']);assert H.blend(FC('You told me your sister lives in Oslo.'),'Where does my sister live?',N,200)[1] and not H.blend(FC('Your sister lives in Rome.'),'Where does my sister live?',N,200)[1]
    os.environ['AMNI_ADAM_T_HARNESS_BLEND']='explain';assert H.blend(FC('You told me your sister lives in Oslo.'),'Where does my sister live?',N,200)==(N['extract'],None);os.environ.pop('AMNI_ADAM_T_HARNESS_BLEND')
    assert H.mat('cs',['ConceptNet (CC BY-SA 4.0): cheese MadeOf milk'],'Cheese is made from milk.')['lines']==['ConceptNet: cheese MadeOf milk']
def test_code_request_detection():
    assert all(K.want(q) for q in ("Write a Python function that reverses a list.","Can you implement insertion sort?","Python function to check if a year is a leap year?","Could you write me a small script that counts words?"))
    assert not any(K.want(q) for q in ("Fix this code: def f(x) return x","What does this code do?","Write a poem about rain.","Write a function in JavaScript that adds two numbers.","What is a Python decorator?"))
@pytest.fixture
def corp(tmp_path,monkeypatch):
    d=tmp_path/'maths';d.mkdir();(d/'triple_value.py').write_text('def triple_value(n: int) -> int:\n    """\n    Return three times n.\n    >>> triple_value(2)\n    6\n    >>> triple_value(5)\n    15\n    >>> triple_value(0)\n    0\n    """\n    return n * 3\n')
    (d/'weird_sum.py').write_text('import os\ndef weird_sum(xs):\n    """\n    Sum a list using a helper.\n    >>> weird_sum([1, 2])\n    3\n    >>> weird_sum([])\n    0\n    """\n    return helper(xs)\n')
    monkeypatch.setattr(K,'CORP',tmp_path);K._IX.clear();yield tmp_path;K._IX.clear()
def test_code_index_keeps_standalone_functions_with_examples(corp):
    X=K.index();assert [r['name'] for r in X['r']]==['triple_value'] and X['r'][0]['ex'][0]==('triple_value(2)','6')
    assert [r['name'] for r in K.find("Write a function that returns three times a value.")]==['triple_value'] and K.find("Write a function that sorts a list of words.")==[]
@JAIL
def test_code_route_composes_and_verifies(corp,on):
    o=K.solve("Write a Python function that returns triple the value of n.");assert o['ok'] and o['how'] in ('composed','retrieved') and K.check(o['code'],o['ex'])
    r=H.run("Write a Python function that returns triple the value of n.",[],'c','c',FC(),160);assert r['slots'][0]['kind']=='code' and r['slots'][0]['ok'] and '```python\ndef triple_value(n' in r['text']
    r=H.run("Write a Python function that sorts a list of words.",[],'c','c',FC(),160);assert r['slots'][0]['hedge'] and '```' not in r['text']
