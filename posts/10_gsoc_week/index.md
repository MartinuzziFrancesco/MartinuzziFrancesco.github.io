# GSoC week 10: Reservoir Memory Machines


For the 10th week of the GSoC program I wanterd to implement a fairly new model, namely the Reservoir Memory Machines (RMM), proposed earlier this year [[1]](#1). This was one of the hardest, and longest, implementations to date and I believe there is still some work to be done. In this post we will briefly touch on the theory behind the model, and after an example of their usage will be presented.

# Theoretical Background
Born as an alternative to the Neural Turing Machine [[2]](#2) the RMM is an extension of the Echo State Network model, with the addition of an actual memory \\( \textbf{M}_t \in \mathbb{R}^{K \times n} \\), a write head and a read head. The dynamics of the RMM are the following:
- In the first step the previous memory state is copied \\( \textbf{M}\_t = \textbf{M}_{t-1} \\), with the initial memory step being initialized to zero. 
- The write head is then controlled by the value \\( c_t^w = \textbf{u}^w \textbf{x}^t + \textbf{v}^r \textbf{h}^t \\) where \\( \textbf{u}^w, \textbf{v}^r \\) are learnable parameters and \\( \textbf{x}^t, \textbf{h}^t \\) are the input vector and state vector at time t respectively. If \\( c_t^w > 0 \\) then the input is written to memory, \\( \textbf{m}\_{t, k} = \textbf{x}^t \\) and \\( k_t = k_{t-1}+1 \\). \\( k \\) is resetted to 1 if it exceeds the memory size \\( K \\). In the other case the memory and \\( k \\) are left as they are.
- Each time step the read head is controlled in a similar way using the vector \\( \textbf{c}\_t^r = \textbf{U}^r \textbf{x}^t + \textbf{V}^r \textbf{h}^t \\) where \\( \textbf{U}^r, \textbf{V}^r \\) are learnable parameters. If \\( c^r_{t, 2} = max{c^r_{t, 1}, c^r_{t, 2}, c^r_{t, 3}} \\) \\( l_t = l_{t-1}+1 \\), otherwise \\( l_t = 1 \\). After that the memory read at time \\( t \\) is set as the \\(  l\\)th row of \\( \textbf{M}\_t \\), \\( \textbf{r}^t = \textbf{m}_{t, l_t} \\)

The output of the system is determined by \\( \textbf{y_t} = \textbf{V} \textbf{h}^t + \textbf{R} \textbf{r}^t \\) where \\( \textbf{V}, \textbf{R} \\) are learnable parameters. Setting \\( \textbf{R} = 0 \\) we can see that the result is the standard ESN.

For a more detailed explanation of the procedure and of the training process please refer to the original paper.

# Implementation in ReservoirComputing.jl

Following both the paper and the code provided (original in Python, click [here](https://gitlab.ub.uni-bielefeld.de/bpaassen/reservoir-memory-machines)) we were able to implement a ```RMM``` mutable struct and a ```RMMdirect_predict``` function able to train and do predictions with the RMM model. The default constructor for ```RMM``` takes as input
- ```W``` the reservoir matrix
- ```in_data``` the training data
- ```out_data``` the desired output
- ```W_in``` the input layer matrix
- ```memory_size``` the size \\( K \\) of the memory
- ```activation``` optional activation function for the reservoir states, with default ```tanh```
- ```alpha``` optional leaking rate, with default 1.0
- ```nla_type``` optional non linear algorithm, eith default ```NLADefault()```
- ```extended_states``` optional boolean for the extended states option, with default ```false```

The constructor trains the RMM, so ance it is initialized there is only need for a predict function. The ```RMMdirect_predict``` takes as input 
- ```rmmm``` an initialized RMM
- ```input``` the input data
and gives as output the prediction based on the input data given. The prediction process is relatively different from the implementation used in ReservoirComputing.jl, so we will not be able to do a proper comparison with the other models we implemented. In the future we do want to uniform the RMM with the other architectures present in the library, but it seems like a moth worth of work, so for the moment we are happy with the basic implementations obtained.

# Examples

For example we will use the next step prediction for the [Henon map](https://en.wikipedia.org/wiki/H%C3%A9non_map), used also in last week test. The map is defined as 

$$x_{x+1} = 1 - ax_n^2 + y_n$$
$$ y_{n+1} = bx_n $$

Let us start by installing and importing all the needed packages:

```julia
using Pkg
Pkg.add("ReservoirComputing")
Pkg.add("DynamicalSystems")
Pkg.add("LinearAlgebra")
```
```julia
using ReservoirComputing
using DynamicalSystems
using LinearAlgebra
```

Now we can generate the Henon map, and we will shift the data points by -0.5 and scale them by 2 to reproduce the data we had last week. The initial transient will be washed out and we will create four datasets called ```train_x```, ```train_y```, ```test_x``` and ```test_y```:`

```julia
ds = Systems.henon()
traj = trajectory(ds, 7000)
data = Matrix(traj)

data = (data .-0.5) .* 2
shift = 200
train_len = 2000
predict_len = 3000
train_x = data[shift:shift+train_len-1, :]
train_y = data[shift+1:shift+train_len, :]

test_x = data[shift+train_len:shift+train_len+predict_len-1, :]
test_y = data[shift+train_len+1:shift+train_len+predict_len, :]
```

Having the needed data we can proceed to the prediction task. In the RMM paper the model is tested using Cycle Reservoirs with Regular Jumps [[3]](#3) so we will do the same for our test. In addition to that we will also use the other minimum complexity reservoirs [[4]](#4) that we implemented in [week 6](https://martinuzzifrancesco.github.io/posts/06_gsoc_week/). The input layer used is obtained with the function ```irrational_sign_input()```, that builds a fully connected layer with the same values which signs are determined by the values of an irrational number, in our case pi. Setting the parameters for the construction of the RMM 

```julia
approx_res_size = 128
sigma = 1.0
beta = 1*10^(-5)
extended_states = false

input_weight = 0.1
cyrcle_weight = 0.99
jump_weight = 0.1
jumps = 12

memory_size = 16
```
We can now build the reservoir and the RMMs needed for the comparison of the results:

```julia
Wcrj = CRJ(approx_res_size, cyrcle_weight, jump_weight, jumps)
Wscr = SCR(approx_res_size, cyrcle_weight)
Wdlrb = DLRB(approx_res_size, cyrcle_weight, jump_weight)
Wdlr = DLR(approx_res_size, cyrcle_weight)

W_in = irrational_sign_input(approx_res_size, size(train_x, 2), input_weight)

rmmcrj = RMM(Wcrj, train_x, train_y, W_in, memory_size, beta)
rmmscr = RMM(Wscr, train_x, train_y, W_in, memory_size, beta)
rmmdlrb = RMM(Wdlrb, train_x, train_y, W_in, memory_size, beta)
rmmdlr = RMM(Wdlr, train_x, train_y, W_in, memory_size, beta)
```

Now that we have our trained RMM we want to predict the one step ahead henon map and compare the results obtained with different reservoirs. In oreder to do so we are first going to implement a quick nmse function :

```julia 
function NMSE(target, output)
    num = 0.0
    den = 0.0
    sums = []
    for i=1:size(target, 2)
        append!(sums, sum(target[:, i]))
    end
    for i=1:size(target, 1)
        num += norm(output[i, :]-target[i, :])^2.0
        den += norm(target[i, :]-sums./size(target, 1))^2.0
    end
    nmse = (num/size(target, 1))/(den/size(target, 1))
    return nmse
end
```

after that we are going to predict the system and compare the results:

```julia 
rmms = [rmmcrj, rmmscr, rmmdlrb, rmmdlr]

for rmm in rmms
    output2 = RMMdirect_predict(rmm, test_x)
    println(NMSE(train_y, output2))
end
```

```
1.4856355716974217
1.4868276240921912
1.5624183281454223
1.5237046076873637
```

As we can see the best performing architecture is the one with the CRJ reservoir. The SCR closely follows. 

This tests are not the one used in the paper, but given that I was a little behind with the implementation I thought to do a couple of quick and easy ones instead. The model is really interesting and I want to continue to explore the possibilities that it offers. The implementation, while working, is not yet finished: there are a couple of finishing touches to give and a couple of more checks to do. I really want to be able to reproduce the results of the paper but the base implementations of ESN in ReservoirComputing and the paper code are really different, and it will take at least another week of full work to unravel all the small details. Huge thanks to the author [Benjamin Paaßen](https://bpaassen.gitlab.io/) that answered quickly and kindly to my emails.

As always, if you have any questions regarding the model, the package or you have found errors in my post, please don’t hesitate to contact me!

## Documentation

<a id="1">[1]</a>
Paaßen, Benjamin, and Alexander Schulz. "Reservoir memory machines." arXiv preprint arXiv:2003.04793 (2020).

<a id="2">[2]</a>
Graves, Alex, Greg Wayne, and Ivo Danihelka. "Neural turing machines." arXiv preprint arXiv:1410.5401 (2014).

<a id="3">[3]</a>
Rodan, Ali, and Peter Tiňo. "Simple deterministically constructed cycle reservoirs with regular jumps." Neural computation 24.7 (2012): 1822-1852.

<a id="4">[4]</a>
Rodan, Ali, and Peter Tino. "Minimum complexity echo state network." IEEE transactions on neural networks 22.1 (2010): 131-144.

