import sys,os,json,urllib.request,subprocess
repo,rev,out=sys.argv[1:4]
os.makedirs(out,exist_ok=True)
m=json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{rev}"))
[subprocess.run(["curl","-sfL","-C","-","-o",os.path.join(out,s["rfilename"]),f"https://huggingface.co/{repo}/resolve/{rev}/{s['rfilename']}"],check=True) for s in m["siblings"] if not s["rfilename"].endswith((".gguf",".ipynb",".sig",".gitattributes")) and "/" not in s["rfilename"]]
print("done",repo,rev,flush=True)
