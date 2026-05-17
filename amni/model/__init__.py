try:from amni.model.layer import TextureLinear
except Exception:TextureLinear=None
try:from amni.model.network import Network
except Exception:Network=None
try:from amni.model.inference import InferenceEngine
except Exception:InferenceEngine=None
__all__=["TextureLinear","Network","InferenceEngine"]
