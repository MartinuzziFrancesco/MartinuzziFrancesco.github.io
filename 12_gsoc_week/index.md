# Google Summer of Code 2020 Final Report


# Introduction

My proposal for the 2020 Google Summer of Code with the Julia Language was based on the implementation of a library for the family of models know as Reservoir Computing. After working for a month on the [issue #34](https://github.com/SciML/NeuralPDE.jl/issues/34) of NeuralPDE.jl I created the initial draft of the ReservoirComputing.jl package, consisting at the time of only the implementation of Echo State Networks (ESNs). Having prior knowledge on the model I started digging a little more on the literature and found a lot of interesting variations of ESNs, which led me to base my proposal entirely on the addition of these models into the existing library. The work done in the three months period was based on weekly goals, which allowed me to keep a steady and consistent pace throughout the project. Most of the implementations done at the end were the one present in the proposal, although there were a couple of little variations from the initial idea. 

# Work done during GSoC
- Week 1: implemented different Linear Model solvers for ESN, tests for the code and checked the results against the literature - [blog post](https://martinuzzifrancesco.github.io/posts/01_gsoc_week/).
- Week 2: implemented Support Vector Echo State Machines (SVESMs), tests for the code and checked the results against the literature - [blog post](https://martinuzzifrancesco.github.io/posts/02_gsoc_week/).
- Week 3: implemented Echo State Gaussian Processes (ESGPs), tests for the code and checked the results against the literature - [blog post](https://martinuzzifrancesco.github.io/posts/03_gsoc_week/).
- Week 4: implemented Singular Value Decomposition-based reservoir, tests for the code and checked the results against the literature - [blog post](https://martinuzzifrancesco.github.io/posts/04_gsoc_week/).
- Week 5: comparison of the various models implemented in the task of predicting chaotic systems both short and long term - [blog post](https://martinuzzifrancesco.github.io/posts/05_gsoc_week/).
- Week 6: implemented minimum complexity ESN methods, tests for the code and checked the results against the literature - [blog post](https://martinuzzifrancesco.github.io/posts/06_gsoc_week/).
- Week 7: implemented Elementary Cellular Automata-based reservoir computers (RECA), tests for the code and checked the results against the literature - [blog post](https://martinuzzifrancesco.github.io/posts/07_gsoc_week/).
- Week 8: implemented two dimensional cellular automata-based reservoir computers, tests for the code and checked the results against the literature - [blog post](https://martinuzzifrancesco.github.io/posts/08_gsoc_week/).
- Week 9: implemented Cycle Reservoirs with Regular Jumps, tests for the code and checked the results against the literature - [blog post](https://martinuzzifrancesco.github.io/posts/09_gsoc_week/).
- Week 10: implemented Reservoir Memory Machines (RMMs), tests for the code and checked the results against the literature - [blog post](https://martinuzzifrancesco.github.io/posts/10_gsoc_week/).
- Week 11: implemented Gated Recurring Unit-based reservoir, tests for the code and checked the results against the literature - [blog post](https://martinuzzifrancesco.github.io/posts/11_gsoc_week/).
- Week 12: added the documentation - [link to documentation](https://reservoir.sciml.ai/dev/).

The code and the library can be found on GitHub, under the SciML organization: [ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl).

# Future directions

I hope to be able to continue to maintain and improve this package in the future, and there are already some ideas of the challenges that could be tackled. Since I started my journey with Julia, my knowledge of the language has increased and looking back there is room for major improvements in the code, both for optimization and usability. Merging all the train and predict function into a single one will surely reduce confusion, and adding more examples to the documentation is already something that I am working on. 

# Acknowledgements

First of all I would like to thanks my mentors Chris, David and Ranjan for following me in this project and for giving me incredible autonomy in these months; a special thanks to Chris that has taken the time to help me from the first commits and all throughout the application process. Another big thanks goes to the Julia community, an incredible group of people that are working towards a more open and welcoming scientific ecosystem, and are always ready to help and guide newcomers as myself; I don't think I have ever felt so welcomed by people I have yet to meet face to face before. Finally thanks to the Google Open Source program for providing both the opportunity to have this experience and the funding to help people partecipate to it.

