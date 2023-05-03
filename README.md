# A simple wireless network communication simulation tool

It is based on an imaginary protocol with a slot time of 100μs, each with a SYNC frame sent by a coordinator node, then a data frame, with specification as noted in the code, and ackknowledgements at the end of a time slot. Using this simplified protocol, it is possible to examine the influence of number of network nodes and user demand with regard to the network throughput.

Run
> python3 simulation.py
