# WhydahGally-LSTM-MLP
Machine Learning Tool with LSTM and MLP Neural Networks, with CUDA implementation.

At the moment there are two MLPs and one LSTM but it is possible to add other machine learning tools.

It is possible to use this tool with CUDA (global and shared memory) or completely without it: switching off the relative preprocessor option.
It's also possible to create mixed configurations: using for certain algorithms CUDA and for others the CPU.
To test the CUDA code you can use this little testing tool I created: https://github.com/Alfiuman/CUDA-Test 
You will be able to see that some (but not all) of the CUDA algorithms present here are slower than an algorithm that uses a good CPU under many conditions, because they mainly move data and don't perform many floating point operations; while algorithms like dot product for matrices are faster in CUDA when they work with large matrices.
But it is important to know that also the CUDA code for dot products for matrices present here can be improved.

The LSTM is mainly based on the design of nicodjimenez. Here the explanation: http://nicodjimenez.github.io/2014/08/08/lstm.html

This code wants to be a basic starting point for ANNs and CUDA, for who wants to learn more about them.
