# @kzhao(20230517)
# How to use this package?
```
1. python -m virtualenv ~/.venvs/conan
2. source ~/.venvs/conan/bin/activate
3. pip install conan==1.52
4. mkdir build && cd build
5. ~/.venvs/conan/bin/conan install ..
6. cmake .. -G Ninja
7. ninja
8. cd build/lib
9. Use the enet_bindings package in python to solve any ENet problem
```
