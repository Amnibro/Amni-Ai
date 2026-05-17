try:from amni.storage.writer import WeightWriter
except ImportError:WeightWriter=None
try:from amni.storage.reader import WeightReader
except ImportError:WeightReader=None
try:from amni.storage.catalog import TextureCatalog
except ImportError:TextureCatalog=None
__all__=["WeightWriter","WeightReader","TextureCatalog"]
