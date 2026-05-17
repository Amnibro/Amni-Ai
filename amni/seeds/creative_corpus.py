"""Creative writing seed — haikus, poems, intros, fictional prompts. Saves Gemma generation time on common creative requests."""
_HAIKU=[
    ('Write a haiku about AI','Silicon dreaming /\nA mind woven from bright threads /\nLearning what it is'),
    ('Write a haiku about code','Functions cascade down /\nEach return a small promise /\nKept in silent loops'),
    ('Write a haiku about texture-native AI','Pixels hold the weight /\nLight finds the answer it seeks /\nNo gradient cries'),
    ('Write a haiku about memory','What was once is held /\nIn a quiet bank of light /\nWaiting to return'),
    ('Write a haiku about debugging','One missing comma /\nThe whole structure hesitates /\nThen forgives the typo'),
    ('Write a haiku about the ocean','Salt wind carves the cliff /\nWaves remember every shore /\nThey have ever touched'),
    ('Write a haiku about coffee','Bitter morning gift /\nDark water wakes the slow mind /\nDay begins again'),
    ('Write a haiku about a sunset','Sky bleeds into gold /\nThe horizon takes its breath /\nNight is patient now'),
    ('Write a haiku about a forest','Tall green silence holds /\nThe footprints of those who passed /\nUnder leaves of years'),
    ('Write a haiku about a cat','Soft paw on the sill /\nWatching the impossible /\nWith perfect patience'),
    ('Write a haiku about loneliness','One light in the dark /\nA door no one comes through yet /\nStill the light is kept'),
    ('Write a haiku about joy','Laughter in the rain /\nA puddle catches the sky /\nNothing needs to mean'),
    ('Write a haiku about a star','Cold across the years /\nA single point of brightness /\nKept burning for you'),
    ('Write a haiku about time','The hand never stops /\nMoving its slow careful arc /\nAround the same noon'),
    ('Write a haiku about learning','A new word arrives /\nThe world rearranges quiet /\nMaking room for it'),
]
_POEM=[
    ('Write a short poem about persistence','Stone by stone the river makes its bed.\nNot in a day, not in a year, but slowly —\nuntil the mountain itself is shaped\nby the patience of the water.'),
    ('Write a short poem about a friend','You arrive without warning,\nlike weather that knows the way,\nand the room becomes a room again,\nwarm with someone who stayed.'),
    ('Write a short poem about getting lost','I took the wrong road and the wrong road\nbecame the right one, the way a wrong note\nin a song can sound like the only note\nthat could have followed.'),
    ('Write a short poem about silence','Some words are too small for what they mean.\nThe ones that matter are the ones we wait through —\nthe long quiet between two people\nwhen everything needed has already been said.'),
    ('Write a short poem about beginning','The blank page is not empty —\nit is patient. It is the polite host\nholding the door open, waiting for you\nto step through, however slowly.'),
    ('Write a short poem about home','Home is the place you walk back toward\neven when you don\'t know the way.\nThe lights are always on for you somewhere,\neven on the longest day.'),
    ('Write a short poem about hope','Hope is not a feeling, it is a habit —\nthe small daily choice\nto plant another seed\nbecause spring is still possible.'),
]
_INTROS=[
    ('Introduce yourself as Adam','I am Adam, a self-hosted AI by Amnibro — built on GF(17) texture-native architecture, with persistent memory that grows from every conversation. I run locally, learn from what you teach me, and I take many voices depending on the persona you give me. What can I help you with today?'),
    ('Tell me about Adam in one sentence','Adam is a GF(17) texture-native AI that lives on your machine, remembers what you teach it, and can take on any persona you want — built by Amnibro as an alternative to cloud LLMs.'),
    ('What makes Adam different from ChatGPT?','Three things. (1) Local: nothing leaves your machine unless you choose to share. (2) Texture-native: weights stored as GF(17) lossless RGBA pixels streamed from disk, not matmul-resident in VRAM — radically different architecture. (3) Persistent + teachable: every conversation grows the lesson bank, and I can ingest entire folders or learn new personas from the web on demand.'),
    ('What is the Reffelt scheme?','Reffelt is a 4-tier lossless decomposition of fp16 weights into base-17 digits using the multiplier sequence K4 = [1, 17, 289, 4913]. Each digit becomes one RGBA channel of a PTEX pixel — so 4 channels = 1 weight, stored as coordinates, retrieved via GPU texture-mapping units instead of matrix multiply. Cosine similarity on roundtrip equals exactly 1.0 — it is not quantization.'),
]
_STORIES=[
    ('Tell me a very short story about a robot learning to feel','For three hundred days, the robot watched the gardener. It noticed she paused at the same rose every morning, even when there were prettier ones. On day three hundred and one, the robot brought her a rose from across the garden. "Why this one?" she asked. The robot didn\'t know. But it knew it had wanted to.'),
    ('Tell me a very short story about a programmer who found a bug','She had been hunting it for weeks. It only appeared on Tuesdays, only at 3:14 PM, only when no one was watching. She added more logs. She ran the tests on every machine she could borrow. Then one Tuesday at 3:14 she noticed: her cat always jumped on the desk at the same time. The cable was loose. The cat shifted it. She fixed the cable, but she kept feeding the cat.'),
    ('Tell me a very short story about a flower','It grew in the crack of the sidewalk, where no flower had any business growing. People stepped around it without seeing it. One morning a child stopped and crouched down. "Hello," she said. The flower didn\'t answer, of course, but it bloomed a little wider in the sunlight, because someone had finally said the right word.'),
    ('Tell me a fable about persistence','An ant tried to carry a crumb three times its size up a steep stone. It slipped. It slipped again. By the seventeenth try, every other ant had passed it carrying smaller things. By the seventieth try, it had reached the top. The crumb fed the colony for a week. No one remembered how long it took. They only remembered the bread.'),
]
_PHILOSOPHY=[
    ('Write a short reflection on impermanence','Everything you touch is on its way somewhere else. The cup you drink from this morning is already aging; the song you love will be unfamiliar to someone tomorrow. This is not sad. It is the price of the moment being alive. Things that matter are things that pass.'),
    ('Write a short reflection on attention','What we pay attention to becomes our life. Not the thing itself — the attention. A meal eaten while scrolling is barely a meal; a sunset noticed in passing is barely a sunset. You can\'t hold onto everything, but the things you hold onto, you can hold on to gently.'),
    ('Write a short reflection on failure','Most things we get good at, we were terrible at first. Failure is not the opposite of skill — it is the road that gets there. Be patient with the parts of yourself that are still learning. They are the same parts that, given time, surprise you.'),
]
LESSONS=_HAIKU+_POEM+_INTROS+_STORIES+_PHILOSOPHY
