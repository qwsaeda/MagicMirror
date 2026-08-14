#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "Building Rust server..."
cd src-server
cargo build --release

echo "Creating distribution package..."
cd ..

mkdir -p out/server

# Copy Rust binary
if [ -f "src-server/target/release/magic-server.exe" ]; then
    cp src-server/target/release/magic-server.exe out/server/server.exe
elif [ -f "src-server/target/release/magic-server" ]; then
    cp src-server/target/release/magic-server out/server/server
fi

# Copy Python inference scripts
cp -r src-python/magic out/server/
cp src-python/server.py out/server/

# Copy models
mkdir -p out/server/models
cp src-python/models/*.onnx out/server/models/

# Create zip
cd out
rm -f server.zip
zip -r server.zip server

echo "Done: out/server.zip"
