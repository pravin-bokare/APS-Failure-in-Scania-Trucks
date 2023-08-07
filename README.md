# **About Dataset**

## **Context**

The dataset consists of data collected from heavy Scania trucks in everyday usage. The system in focus is the Air Pressure system (APS) which generates pressurized air that is utilized in various functions in a truck, such as braking and gear changes. The datasets' positive class consists of component failures for a specific component of the APS system. The negative class consists of trucks with failures for components not related to the APS. The data consists of a subset of all available data, selected by experts.

## **Content**

The training set contains 60000 examples in total in which 59000 belong to the negative class and 1000 positive class. The test set contains 16000 examples. There are 171 attributes per record.

The attribute names of the data have been anonymized for proprietary reasons. It consists of both single numerical counters and histograms consisting of bins with different conditions. Typically the histograms have open-ended conditions at each end. For example, if we measuring the ambient temperature "T" then the histogram could be defined with 4 bins where:

The attributes are as follows: class, then anonymized operational data. The operational data have an identifier and a bin id, like "Identifier_Bin". In total there are 171 attributes, of which 7 are histogram variables. Missing values are denoted by "na".

## Acknowledgements

This file is part of APS Failure and Operational Data for Scania Trucks. It was imported from the UCI ML Repository.
    

# **Dataset Description**

This program (APS Failure and Operational Data for Scania Trucks) is 
free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.



------------------------------------------------------------------------

1. Title: APS Failure at Scania Trucks


2. Source Information
   -- Creator: Scania CV AB
               Vagnmakarv�gen 1 
               151 32 S�dert�lje 
               Stockholm
               Sweden 
   -- Donor:   Tony Lindgren (tony@dsv.su.se) and Jonas Biteus (jonas.biteus@scania.com)
   -- Date:    September, 2016
 

3. Past Usage:
   Industrial Challenge 2016 at The 15th International Symposium on Intelligent Data Analysis (IDA) 
   -- Results:         
     The top three contestants                                                | Score | Number of Type 1 faults | Number of Type 2 faults
     ------------------------------------------------------------------------------------------------------------------------------------
     Camila F. Costa and Mario A. Nascimento                                  | 9920  | 542                     | 9
     Christopher Gondek, Daniel Hafner and Oliver R. Sampson                  | 10900 | 490                     | 12
     Sumeet Garnaik, Sushovan Das, Rama Syamala Sreepada and Bidyut Kr. Patra | 11480 | 398                     | 15


4. Relevant Information:
   -- Introduction
     The dataset consists of data collected from heavy Scania 
     trucks in everyday usage. The system in focus is the 
     Air Pressure system (APS) which generates pressurised 
     air that are utilized in various functions in a truck, 
     such as braking and gear changes. The datasets' 
     positive class consists of component failures 
     for a specific component of the APS system. 
     The negative class consists of trucks with failures 
     for components not related to the APS. The data consists 
     of a subset of all available data, selected by experts.


5. Number of Instances: 
     The training set contains 60000 examples in total in which 
     59000 belong to the negative class and 1000 positive class. 
     The test set contains 16000 examples. 

6. Number of Attributes: 171 


7. Attribute Information:
   The attribute names of the data have been anonymized for 
   proprietary reasons. It consists of both single numerical 
   counters and histograms consisting of bins with different 
   conditions. Typically the histograms have open-ended 
   conditions at each end. For example if we measuring 
   the ambient temperature "T" then the histogram could 
   be defined with 4 bins where: 


  The attributes are as follows: class, then 
  anonymized operational data. The operational data have 
  an identifier and a bin id, like "Identifier_Bin".
  In total there are 171 attributes, of which 7 are 
  histogram variabels. Missing values are denoted by "na".
