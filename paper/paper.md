---
title: 'SenAOReFoc: A Closed-Loop Sensorbased Adaptive Optics and Remote Focusing Control Software'
tags:
  - Python
  - Sensorbased
  - Adaptive optics
  - Remote focusing
  - DM
  - SHWS
authors:
  - name: Jiahe Cui^[jiahe.cui\@eng.ox.ac.uk]
    orcid: 0000-0003-2122-5754
    affiliation: 1
  - name: Karen M. Hampson
    affiliation: 1
  - name: Matthew Wincott
    affiliation: 1
  - name: Martin J. Booth
    affiliation: 1
affiliations:
  - name: Department of Engineering Science, University of Oxford, Parks Road, Oxford, OX1 3PJ, UK
    index: 1
date: 16 December 2021
bibliography: paper.bib

---

# Summary

SenAOReFoc is a closed-loop sensorbased adaptive optics (AO)
and remote focusing control software that works with a deformable mirror
(DM) and a Shack-Hartmann wavefront sensor (SHWS). It provides a 
user-friendly graphic user interface (GUI) with modular widget arrangements
and clear labelling to help the user navigate through different software 
functionalities. Interactive messages are also displayed from the GUI for 
user guidance.

SenAOReFoc consists of 5 main units, the SHWS initialisation and DM
calibration unit, the Zernike aberration input unit, the AO control and
data collection unit, the miscellaneous control unit, and the remote
focusing unit, as shown in \autoref{fig:gui}. The software can be ran in either 'debug mode' to perform 
functionality tests without connected hardware (DM and SHWS), or 
'standard mode' on a well-aligned optical sectioning microscope (confocal, 
multiphoton, etc.). User controllable system parameters can be freely accessed and modified
in a separate configuration file that is loaded upon software initialisation, 
and parameters that require continuous user input can be modified from the GUI. Parameters calculated when running 
the software, as well as important result data, are grouped and saved in a separate
HDF5 file that can be read with HDFView software. Automated AO performance characterisations can be
performed in 'standard mode' to assess the correction ability of the optical 
system. If the adopted DM is designed with a large stroke, i.e., is 
capable of large deformations, both the closed-loop AO correction and 
remote focusing functionalities can be exploited. On the other hand, if the 
DM exhibits insufficient stroke for remote focusing, by ignoring the remote 
focusing unit, closed-loop AO correction functionalities will still be fully 
functional without additional modifications to the software. 

Closed-loop AO correction can be performed using both the zonal method,
which updates DM control voltages in terms of the raw slope values; and
the modal method, which updates DM control voltages in terms of
orthogonal Zernike polynomials. There are four sub-modes tagged to each
of the two methods: 1) standard closed-loop AO correction; 2)
closed-loop AO correction with consideration of obscured search blocks;
3) closed-loop AO correction with partial correction excluding defocus; 
and 4) closed-loop AO correction with both consideration of obscured 
search blocks and partial correction excluding defocus.

Remote focusing can be performed by scanning the focus axially with a
pre-determined axial range, step increment and step number, or by
manually adjusting a toggle bar on the GUI for random access remote
focusing. The former also incorporates options of whether or not to
perform closed-loop AO correction at each remote focusing depth.

![Graphic user interface (GUI) of SenAOReFoc with modular units labelled in red boxes.\label{fig:gui}](media/image1.png){ width=100% }

# Statement of need

The performance of optical microscopes degrades significantly in the
presence of optical aberrations that develop due to misalignments in the
optical system and inhomogeneities in refractive index of the imaging
sample. This results in compromised image resolution and contrast.
Adaptive optics (AO) is a powerful technique that corrects for optical
aberrations using reconfigurable adaptive devices, such as a deformable
mirror (DM) or spatial light modulator (SLM), to restore the image
quality [@Booth2014LightScience&Applications]. This is most crucial for
high-quality imaging, such as super-resolution or deep imaging.

Remote focusing is an axial scanning technique recently introduced to 
avoid slow movements of the sample stage and objective lens during real-time
applications [@Ji2016NatureNeuroscience]. By using a DM for remote 
focusing, calibration can be performed through closed-loop AO correction to
simultaneously correct for system aberrations introduced by beam divergence 
at different focusing depths [@Zurauskas2017BiomedicalOpticsExpress].

Sensorbased AO, which uses a dedicated sensor to measure the wavefront 
[@Platt2001JournalofRefractiveSurgery], was first introduced in the field of astronomy [@Beckers1993]. 
Since then, it has also been widely adopted in vision science [@Porter2006; 
@Godara2010Optometryandvisionscience] and microscopy [@Booth2007PhilosophicalTransactions; 
@Ji2017NatureMethods]. A range of software serving for different astronomical 
purposes have been introduced over time, including those for simulation of AO
systems [@Carbillet2005MonthlyNoticesoftheRoyalAstronomicalSociety; @Conan2014], control with
atmospheric tomography [@Ahmadia2005], and fast control using GPU hardware 
[@Guyon2018]. In the microscopy community, software is also openly available for AO modelling and
analysis [@Townson2019OpticsExpress], as well as general AO control [@Hall2020OpticsExpress]. 
However, there is not yet an existing open-source software that serves the purpose of
performing both AO and remote focusing within the single package, despite that the techniques 
have been widely adopted. As a result, time and effort has to be spent reimplementing existing techniques 
for different hardware systems. 

SenAOReFoc aims to fill this gap and to make closed-loop sensorbased AO [@Fernandez2001OpticsLetters] and 
remote focusing more easily accessible to the microscopy community. It has been
designed for open development and easy integration into existing adaptive optical
microscopes with a simple and user-friendly architecture. The functionality of the software has also been tested
on different operating systems (Windows/Linux/macOS) for sake of generality. The software fully operates in 
Windows under both 'debug mode' and 'standard mode' for hardware control. For Linux and macOS, functionality tests
in 'debug mode' passes for both systems without loading hardware packages. Python-bindings for Ximea cameras
and Python 3.8-binding for Alpao DMs are compatible with Linux systems, and other Python-bindings for Alpao DMs
should also be compatible following their release. However, we note that SenAOReFoc is a control software, and the 
performance of the closed-loop AO correction in practice is inevitably dependent on the state of the optical system.
Finally, SenAOReFoc takes care of some reported issues in the field, such as the thermal effects of electromagnetic DMs 
[@Bitenc2017OpticsExpress], and obscured search blocks in the case of severely distorted
pupils [@Dong2018OpticsExpress; @Ye2015OpticsExpress; @Cui2020Zenodo]. In the former case, an option is provided to exercise
the DM membrane by sending routine sets of random control voltages to the actuators upon software initialisation. In the latter
case, control algorithms for both the zonal and modal methods are modified as compared to the standard case to ensure 
the orthogonality of control. 

# Example usage

Four examples are given for the usage of this software, the first three for automated AO performance characterisations
of the reflectance confocal microscope described in [@Cui2021Biophotonics; @Cui2021OpticsExpress]; and the last for 
remote focusing in frozen mouse skull as reported in [@Cui2021Biophotonics].

**Example 1**: The dynamic range of the SHWS for the first *N* Zernike modes can be 
characterised by generating and correcting for them in closed-loop using mode 0/1 of 
[Data Collection]. \autoref{fig:example1}(a) provides an example when characterising Zernike modes 
3-20 (excl. tip/tilt). \autoref{fig:example1}(b)-(e) provide examples of generated and detected RMS 
amplitudes of selected odd/even Zernike modes in increments of 0.02 micrometers. Parameters 
in `config.yaml` under `AO` and `data_collect` were set as follows.

```python
AO:
control_coeff_num: 20  # Number of zernike modes to control during AO correction

data_collect:
  data_collect_mode: 1  # Mode flag for Data Collection button, detailed function descriptions in app.py
  loop_max_gen: 15  # Maximum number of loops during closed-loop generation of zernike modes in mode 0/1/2/3
  incre_num: 15  # Number of zernike mode amplitudes to generate in mode 0/1
  incre_amp: 0.02  # Increment amplitude between zernike modes in mode 0/1
  run_num: 5  # Number of times to run mode 0/1/2/3
```

![Characterisation results of system AO correction performance.
(a) Dynamic range of the SHWS for Zernike modes 3∼20 (excl. tip/tilt).
(b)-(e) Generated and detected RMS amplitudes of odd/even Zernike modes
(b) 5, (c) 7, (d) 12, and (e) 20, in increments of 0.02 micrometers. 5 tests were
performed for each measurement.\label{fig:example1}](media/image2.png){ width=60% }

**Example 2**: The degree of Zernike mode coupling upon detection at the
SHWS can be characterised by individually generating the same amount of
each mode on the DM and constructing a heatmap of the detected Zernike
coefficients using mode 0/1 of [Data Collection]. \autoref{fig:example2} provides an example heatmap of correlation coefficients 
between detected and generated mode values for 0.1 micrometers of Zernike modes 3-20 (excl. tip/tilt). Parameters in
`config.yaml` under `data_collect` were set as follows.

```python
data_collect:
  data_collect_mode: 1  # Mode flag for Data Collection button, detailed function descriptions in app.py
  loop_max_gen: 15  # Maximum number of loops during closed-loop generation of zernike modes in mode 0/1/2/3
  incre_num: 1  # Number of zernike mode amplitudes to generate in mode 0/1
  incre_amp: 0.1  # Increment amplitude between zernike modes in mode 0/1
  run_num: 5  # Number of times to run mode 0/1/2/3
```

![Heatmap of correlation coefficients between detected and
generated mode values for 0.1 micrometers of Zernike modes 3-20 (excl. tip/tilt).\label{fig:example2}](media/image3.png){ width=60% }

**Example 3**: To ensure the system can correct for multiple Zernike
modes with good stability and minimal mode coupling, different
combinations of odd and even Zernike modes can be generated and
corrected for in closed-loop using mode 2/3 of [Data Collection]. \autoref{fig:example3} provides an example of the detect amplitude
and Strehl ratio after each closed-loop iteration for some odd and 
even Zernike mode combinations. Parameters in `config.yaml` under `data_collect` were set as follows.

```python
data_collect:
  data_collect_mode: 3  # Mode flag for Data Collection button, detailed function descriptions in app.py
  loop_max_gen: 15  # Maximum number of loops during closed-loop generation of zernike modes in mode 0/1/2/3
  run_num: 5  # Number of times to run mode 0/1/2/3
```

And [Zernike array edit box] was set to [0 0 0 0 **0.1** 0 **0.1**
0 0 0 0 **0.1** 0 0 0 0 0 0 0 **0.1**].

![Detected amplitudes of generated odd and even Zernike mode
combinations and Strehl ratio calculated using first 69 Zernike modes
(excl. tip/tilt) after each closed-loop iteration. 5 tests were performed for each measurement.
\label{fig:example3}](media/image4.png)

**Example 4**: Voltages that control DM actuators to deform the membrane for fine 
axial remote focusing was calibrated according to the procedure explained in the GitHub repository 
<https://github.com/jiahecui/SenAOReFoc>, and reported in [@Cui2021Biophotonics] and [@Cui2021OpticsExpress]. 
[Calibrate] was pressed for both directions of the optical axis and a piece of white card was displaced by 
10 micrometers each time before pressing 'y' on the keyboard to proceed with closed-loop
AO correction. Interpolation of the DM control voltages at each calibration step was then performed 
to obtain those for 0.1 micrometer increments. The software's remote focusing capability was 
demonstrated in frozen mouse skull in [@Cui2021Biophotonics] to show no noticeable difference in
resolution and size of field of view as compared to standard translation of the sample stage. 
Results are also shown in \autoref{fig:example4}.

![Remote focusing results within frozen mouse skull. (a) Stack images acquired with 10 micrometer
axial steps over a 90 micrometer depth range. Top rows: results obtained by scanning sequentially 
through multiple depths using precalibrated DM control voltages. Bottom rows: comparison results 
obtained by axially translating the sample stage only. (b) Colour-coded maximum intensity projection
of 90 frames with 1 micrometer intervals. Scale bar: 100 micrometers.\label{fig:example4}](media/image5.png)

# How to cite

If you're using any modified version of this software for your work, please cite this paper. If you're
performing remote focusing using the calibration procedure set out in this software, please also cite
either [@Cui2021Biophotonics] or [@Cui2021OpticsExpress].

# Acknowledgements

We thank Chao He for testing the software functionality using different operating systems. 
This work was supported by European Research Council projects 695140 and 812998.

# References
