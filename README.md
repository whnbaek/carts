# CARTS - Compiler for Asynchronous Runtime System

CARTS is an LLVM/MLIR-based compiler framework that implements the ARTS (Asynchronous Runtime System) dialect for distributed programming on modern architectures.

## Full Documentation (Notion)
- [CARTS Documentation](https://www.notion.so/248fde4bf3a2816d8c11f3c1cf5c4617)
- [Architecture & Dependencies](https://www.notion.so/249fde4bf3a281b5ab16e087ee4056b1)
- [Build and Run](https://www.notion.so/248fde4bf3a281dca4dde3c5504313c5)
- [Compiler Pipeline](https://www.notion.so/248fde4bf3a2812ea12bc52421240543)
- [Runtime Integration](https://www.notion.so/249fde4bf3a2813ea115d9731d09706e)
- [Next Steps Dashboard (Kanban)](https://www.notion.so/a5b6038e1d06457ba2c97032d0da8e79)

## Quick Start

```bash
# Install dependencies, build, and setup PATH automatically
python3 tools/setup/carts-setup.py

# Build project
carts build

# Complete pipeline from C++ to executable
carts execute simple.cpp -o simple
```

Notes:
- Always use the `carts` wrapper for all operations (conversion, optimization, lowering, compile/link).
- Project build uses system clang; ARTS operations use the installed LLVM toolchain.
- The setup script automatically adds `carts` to your PATH.

## Contributing
- Track work in the [Next Steps Dashboard](https://www.notion.so/a5b6038e1d06457ba2c97032d0da8e79)

---

**Author**: Rafael Andres Herrera Guaitero  
**Email**: <rafaelhg@udel.edu>