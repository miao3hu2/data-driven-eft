Please check notebooks for a quick demonstration of demo. I recommand to read 02_llm_pipeline, as this is the only notebook that is rewritten by human. 

All codes in src/core has been carefully checked. The others in src have been skimmed. More detailed instruction is on the way. 

I also welcome collaborations with people who specalize in any one of the following fields: dynamical systems, stochastic differential equations, computational fluid dynamics, machine learning algorithms, turbulence engineerings, ...

Currently I have just made the whole pipeline okay to run. There are many simplifications in the code. Feel free to leave comments. 

Known limits and questions: 
1. BlockCutoffFilter always drops the Nyquist mode
2. The time step for inferring Mori Kernel is always 1
3. The definition of the inner product between operators must be carefully chosen according to the measure in the dynamic process.
4. The current RL framework may be replaced by others.
5. StochasticRNN may be also replaced by some other structures of NN with stochastics. 
6. The balance between the accuracy, speed, and penalty must be carefully taken for designing a better reward. 
7. Other RG flow based reward functions
8. RG flow in demo is assumed to be a straight line to the extrapolate level, in general, this cannot be true. Consider to use a diffusion model. 
9. There is a repeated computation in getting the operator_shares. For now, since the reward function is temporary, we keep the current version. 
10. Currently, the llm will understand Du as the x direction derivative of u and u is only supported on 1D lattice. 
11. Also, the model built for capturing orthogonal dynamics support 1D lattice only. 
12. Currently, the demo are tested on periodic boundary conditions. We may need other methods to treat a more general boundary conditions. 
13. The network for orthogonal_dynamics may also change depending on need. 