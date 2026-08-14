python -m nuitka --standalone --assume-yes-for-downloads \
  --include-data-files="src-python/models/*.onnx=models/" \
  --output-dir=out src-python/server.py

cd out/server.dist && zip -r ../server.zip .

echo "✅ Done"