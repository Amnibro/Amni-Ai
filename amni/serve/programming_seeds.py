"""Curated programming-knowledge seeds for Adam's head-start crawl.
Three source kinds, all funneled through the existing ingest pipeline (_safe_fetch -> _distill ->
extract_qa_pairs -> teach) into the routed map-PTEX store (partitioned by language/topic):
  1. PROGRAMMING_TOPICS  — DDG search phrases that surface GitHub/StackOverflow/docs.
  2. CANONICAL_SOURCES   — direct high-value MIT/Apache/public-doc URLs (awesome-lists, TheAlgorithms, refs).
  3. HF_CODE_DATASETS    — permissively-licensed HuggingFace code datasets (for direct sample ingestion, I7).
Run on-demand + bounded via run_programming_bootstrap()/bootstrap_all()."""

_LANGS=['python','rust','javascript','typescript','go','java','c','c++','c#','ruby','php','swift','kotlin','scala','haskell','elixir','clojure','bash','sql','r','julia','lua','perl','zig','dart','ocaml','erlang','fsharp','nim','crystal']
_CORE_TOPICS=['core syntax and idioms','data structures','error handling best practices','common bugs and fixes','testing and assertions','performance and optimization','concurrency and parallelism','file and IO','standard library essentials','design patterns']
PROGRAMMING_TOPICS=[f'{lang} {topic} examples' for lang in _LANGS for topic in _CORE_TOPICS]+[
    'binary search implementation explained github',
    'quicksort mergesort heapsort implementation comparison',
    'hash table data structure implementation',
    'linked list tree graph operations implementation',
    'dynamic programming common patterns examples',
    'big O time complexity common algorithms cheatsheet',
    'recursion vs iteration tradeoffs examples',
    'git rebase merge cherry-pick workflow explained',
    'rest api design best practices',
    'sql query optimization indexes best practices',
    'regular expressions cross-language cheatsheet',
    'data structures and algorithms interview cheatsheet github',
    'clean code principles refactoring patterns',
    'unit integration test pyramid best practices',
    'concurrency primitives mutex semaphore channel comparison',
]
# Direct high-value, permissively-licensed sources (MIT/Apache/CC). (label, language, url)
CANONICAL_SOURCES=[
    ('TheAlgorithms/Python (MIT)','python','https://raw.githubusercontent.com/TheAlgorithms/Python/master/DIRECTORY.md'),
    ('TheAlgorithms/Java (MIT)','java','https://raw.githubusercontent.com/TheAlgorithms/Java/master/DIRECTORY.md'),
    ('TheAlgorithms/JavaScript (MIT)','javascript','https://raw.githubusercontent.com/TheAlgorithms/JavaScript/master/DIRECTORY.md'),
    ('TheAlgorithms/Go (MIT)','go','https://raw.githubusercontent.com/TheAlgorithms/Go/master/DIRECTORY.md'),
    ('TheAlgorithms/Rust (MIT)','rust','https://raw.githubusercontent.com/TheAlgorithms/Rust/master/DIRECTORY.md'),
    ('TheAlgorithms/C-Plus-Plus (MIT)','c++','https://raw.githubusercontent.com/TheAlgorithms/C-Plus-Plus/master/DIRECTORY.md'),
    ('awesome-python (CC)','python','https://raw.githubusercontent.com/vinta/awesome-python/master/README.md'),
    ('awesome-rust (CC0)','rust','https://raw.githubusercontent.com/rust-unofficial/awesome-rust/master/README.md'),
    ('awesome-go (MIT)','go','https://raw.githubusercontent.com/avelino/awesome-go/main/README.md'),
    ('awesome-javascript (MIT)','javascript','https://raw.githubusercontent.com/sorrycc/awesome-javascript/master/README.md'),
    ('awesome-cpp (MIT)','c++','https://raw.githubusercontent.com/fffaraz/awesome-cpp/master/README.md'),
    ('awesome-java (CC)','java','https://raw.githubusercontent.com/akullpp/awesome-java/master/README.md'),
    ('awesome-scala (CC)','scala','https://raw.githubusercontent.com/lauris/awesome-scala/master/README.md'),
    ('awesome-haskell (MIT)','haskell','https://raw.githubusercontent.com/krispo/awesome-haskell/master/README.md'),
    ('awesome-elixir (CC)','elixir','https://raw.githubusercontent.com/h4cc/awesome-elixir/master/README.md'),
    ('awesome-swift (MIT)','swift','https://raw.githubusercontent.com/matteocrippa/awesome-swift/master/README.md'),
    ('awesome-kotlin (Apache)','kotlin','https://raw.githubusercontent.com/KotlinBy/awesome-kotlin/master/readme.md'),
    ('awesome-design-patterns (CC0)','general','https://raw.githubusercontent.com/DovAmir/awesome-design-patterns/master/README.md'),
    ('awesome-algorithms (CC)','general','https://raw.githubusercontent.com/tayllan/awesome-algorithms/master/README.md'),
    ('build-your-own-x (CC0)','general','https://raw.githubusercontent.com/codecrafters-io/build-your-own-x/master/README.md'),
]
# Permissively-licensed HF code datasets (direct sample ingestion — I7). Filter to MIT/Apache/BSD at pull time.
HF_CODE_DATASETS=['bigcode/the-stack-dedup','bigcode/the-stack-smol','codeparrot/codeparrot-clean-valid','codeparrot/github-code-clean','code_search_net']
