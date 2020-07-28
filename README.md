# Dynasim

Some short documentation regarding the code used for work. 

## Goal
We have a sample from the SOEP where people drop out after some time and new ones are added. To "fill up" the people who drop out we use a dynmic microsimulation. This model aims to caputre the lifecycle of a person and is then used to impute the missing people.

In additon to standard regression techniques we also use the _sklearn_ library to implement some ML procedures and then compare them to the usual approaches.

## Structure of Code

Concerning the structure of the model we mostly follow the literature and split it into two modules. In each year of the simulation both modules are run. The following functions are included in them respectively. 

### Family module

* Death: 
  Given their age people die with higher and higher probabilities. As soon as they reach a maximum age (right now 99 years) they die for sure.
* Separations:
  Couples separate with a certain probability. we assume that the male partner always moves out and that the children stay with the mother. Because of this we do not consider same-sex couples right now. The percentage is pretty low and these assumptions make it compuationally easier to implement.
* Marriages:
  Couples marry with a certain probability.
* Dating Market:
  Based on personal characteristics such as age, education and earnings single people find a new partner with a given probability.
* Birth:
  Women aged between 15 and 49 give birth to one child with a probability depending on ehir age.

### Work module
Until now we estimate the different employment status one after the other following the literatuere. Using classifier algorithms this can be done in one step later.

* Labor Force Participation (binary)
* Working (binary); conditional on being in the labor force
* Fulltime (binary); conditional on working
* Hours (continuous); conditional on working
* Wage (continuous); conditional on working
