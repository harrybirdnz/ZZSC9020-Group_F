// Document Setup
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set page(margin: 110pt)
#set text(region: "AU", size:10pt, font: "New Computer Modern")

// Heading Styles (to match template)
#show heading: it => [#text(weight: "regular")[#it]]
#let head(body) = {
  set align(center)
  set text(size: 14pt, weight: "regular")
  [#line(length: 100%)
  #body \
  #line(length: 100%)\ ]
}
#show heading.where(level: 1, outlined:true).and(
  heading.where(body:[Introduction])).or(
  heading.where(body:[Literature Review])).or(
  heading.where(body:[Material and Methods])).or(
  heading.where(body:[Exploratory Data Analysis])).or(
  heading.where(body:[Analysis and Results])).or(
  heading.where(body:[Discussion])).or(
  heading.where(body:[Conclusion and Further Issues])): it => [
  #head[
    #smallcaps[Chapter] #counter(heading).display()\
    #it.body
]
]
// Table Caption Above
#show figure.where(
  kind: table
): set figure.caption(position: top)

// Begin Document
#place(
  top + center,
  scope: "parent",
  float: true,
  text(1.1em)[
#image("media/unsw-logo.png", height: 50pt)
#v(10%)
#text(size: 15pt)[
  CAPSTONE PROJECT BY GROUP F\
A DATA SCIENCE APPROACH TO FORECAST ELECTRICITY CONSUMPTION IN AUSTRALIA
]
#v(20%)
Cameron Botsford (z5496223)\
Harry Bird (z5579579)\
Nidhi Wadhwa (z5536904)\
Saba Abbaslou (z5380391)\
#v(5%)
School of Mathematics and Statistics\
UNSW Sydney\
#v(5%)
September 2025\
#v(20%)
#align(bottom)[#smallcaps()[Submitted in partial fulfilment of the requirements of
the capstone course ZZSC9020]]
  ],
)

#pagebreak()
#counter(page).update(1)
#set page(numbering: "i")
#head[Plagiarism statement\ ]
\
I declare that this thesis is my own work, except where acknowledged, and has not been submitted for academic credit elsewhere. \
\
I acknowledge that the assessor of this thesis may, for the purpose of assessing it:
- Reproduce it and provide a copy to another member of the University; and/or,
- Communicate a copy of it to a plagiarism checking service (which may then retain a copy of it on its database for the purpose of future plagiarism checking).\
\
I certify that I have read and understood the University Rules in respect of Student Academic Misconduct, and am aware of any potential plagiarism penalties which may apply.\
\
By signing this declaration I am agreeing to the statements and conditions above.\
\
Signed: #box(height: 0pt)[#line(length: 45%)] #h(10%) Date: #box(height: 0pt)[#line(length: 25%)]\
#v(20pt)
Signed: #box(height: 0pt)[#line(length: 45%)] #h(10%) Date: #box(height: 0pt)[#line(length: 25%)]\
#v(20pt)
Signed: #box(height: 0pt)[#line(length: 45%)] #h(10%) Date: #box(height: 0pt)[#line(length: 25%)]\
#v(20pt)
Signed: #box(height: 0pt)[#line(length: 45%)] #h(10%) Date: #box(height: 0pt)[#line(length: 25%)]\

#pagebreak()
#head[Acknowledgements]
By far the greatest thanks must go to my supervisor for the guidance, care and support they provided.\ \
Thanks must also go to Emily, Michelle, John and Alex who helped by proof-reading the document in the final stages of preparation.\ \
Although I have not lived with them for a number of years, my family also deserve many thanks for their encouragement. Thanks go to Robert Taggart for allowing
his thesis style to be shamelessly copied.

#pagebreak()
#head[Abstract]
The image below gives you some hint about how to write a good abstract.
#image("media/abstract.png")

#pagebreak()
#outline(title: [#head[Contents]])

#pagebreak()
#counter(page).update(1)
#set page(numbering: "1")
= Introduction
This Template can be used for the ZZSC9020 course report. We suggest you organise your report using the following chapters but, depending on your own project, nothing prevents you to have a different organisation.

#pagebreak()
= Literature Review
See #link("https://typst.app/docs/reference/model/ref/") for how to reference sources and figures.\ \
In order to incorporate your own references in this report, we strongly advise you use BibTeX. Your references then needs to be recorded in the file references.bib.\ \
Typst also supports the Hayagriva .yml format which I find is a lot easier to read than .bib, however most sources let you export directly as .bib so using the .yml requires conversion (https://jonasloos.github.io/bibtex-to-hayagriva-webapp/).\
Here is a reference using .yml @aemo1.\
Here is a reference using .bib @google1.\
Both show up in the references so use whatever you prefer.

== Model Specific
=== Linear Regression
Saba to complete
=== Convolutional Neural Network
Nidhi to complete
=== Long Short-Term Memory Network
Cameron to complete
=== Transformer
The Transformer network architecture was introduced in 2017 by researchers at Google @google1. It was designed to replace and outperform the recurrence based models used at the time, both in increased performance and reduced training cost due to parallelisation @transformer2. The architecture is a specific instance of the encoder-decoder models that had become popular in the years prior @transformer1. The primary advancement from this architecture was in the space of natural language processing (NLP), with a myriad of models being developed and becoming familiar in the mainstream such as ChatGPT @transformer1. However, this architecture can also still be applied to forecasting problems, and has been successfully @transformer2. 


#pagebreak()
= Material and Methods
== Software
R and Python of course are great software for Data Science. Sometimes, you might
want to use bash utilities such as awk or sed.
Of course, to ensure reproducibility, you should use something like Git and
RMarkdown (or a Jupyter Notebook). Do not use Word!
== Description of the Data
#emph[How are the data stored? What are the sizes of the data files? How many files? etc.]
== Pre-processing Steps
#emph[What did you have to do to transform the data so that they become useable?]
- Missing Values
- Irregular Timesteps
- Augmentation
=== totaldemand_nsw
This file contains 22 entries where datetime is null. These must be removed.
There are 11 instances where 3 records have identical datetimes - all at 3am??? Daylight savings - need to document this better. There were no entries with valid datetime that had null demand.

=== temperature_nsw.csv
There are three missing days. Why???
How should we deal with this?

=== Heating Degree Days (HDD) and Cooling Degree Days (CDD)
HDD and CDD are variables that are used to measure heating and cooling requirements. This estimate is based on the difference between the air temperature and a critical temperature set by AEMO. For New South Wales, the HDD critical temperature is 17.0 degrees C and the CDD critical temperature is 19.5 degrees C @aemo1. @hdd calculates the HDD and @cdd calculates CDD.
$ "HDD" = "Max"(0, 17 - overline(T)) $ <hdd>
$ "CDD" = "Max"(0, overline(T) - 19.5 ) $ <cdd>

== Data Cleaning
How did you deal with missing data? etc.
== Assumptions
What assumptions are you making on the data?
== Modelling Methods
=== Linear Regression
Saba to complete
=== Convolutional Neural Network
Nidhi to complete
=== Long Short-Term Memory Network
Cameron to complete
=== Transformer
The transformer takes an input, which in NLP is a sentence or phrase, that is first converted to numbers by an embedding layer before being passed to the encoder portion of the transformer @transformer2. Sequentiality is captured through positional encoding. In our task, we aim to input sequential demand and temperature data, and output a prediction for the next 24 hours of electricity demand. 
==== Input
The input can be further broken down between historical data and contextual data. Historical data is the actual temperature and demand recordings. Contextual data is that which can be extracted from the date/time, such as day of the week and month of the year. 

#pagebreak()
= Exploratory Data Analysis
#emph[This is where you explore your data using histograms, scatterplots, boxplots, numerical summaries, etc.] 

#figure(
  image("media/weekday_demand.png"),
  caption: [Daily Average Demand]
)
#figure(
  image("media/month_demand.png"),
  caption: [Monthly Average Demand]
)
#figure(
  image("media/seasonal_demand.png"),
  caption: [Seasonal Average Demand]
)
#figure(
  image("media/time_series.png"),
  caption: [Time Series Exploration]
)
#figure(
  image("media/seasonal_comparison.png"),
  caption: [Seasonal Comparison]
)
#figure(
  image("media/scatterplots.png"),
  caption: [Scatterplots]
)
#figure(
  image("media/scatterplots2.png"),
  caption: [Scatterplots]
)
#figure(
  image("media/scatterplots3.png"),
  caption: [Scatterplots]
)
#figure(
  image("media/scatterplots5.png"),
  caption: [Scatterplots]
)
#figure(
  image("media/correlation.png"),
  caption: [Correlations]
)





#pagebreak()
= Analysis and Results
== A First Model
Having a very simple model is always good so that you can benchmark any result
you would obtain with a more elaborate model.\
For example, one can use the linear regression model
$ Y_i = beta_0 + beta_1x_(1i) + ... + beta_p x_(pi) + epsilon_i, i=1,...,n $

where it is assumed that the $epsilon_i$’s are i.i.d. N(0, 1).

== Accuracy
Accuracy of each model was determined using mean absolute percentage error (MAPE), defined in @mape.
$ "MAPE" = 1/N sum_(i=1)^N abs(y_i - hat(y)_i)/y_i $ <mape>

#pagebreak()
= Discussion
Put the results you got in the previous chapter in perspective with respect to the
problem studied.

#pagebreak()
= Conclusion and Further Issues
What are the main conclusions? What are your recommendations for the “client”?
What further analysis could be done in the future?

#pagebreak()
#head[References]
#{
  show heading: none
  bibliography(("references.bib", "bib.yml"), style: "ieee", title: [References])
}
#pagebreak()
#set heading(numbering: none)
#head[Appendix]
#{
  show heading: none
  heading(numbering: none)[Appendix]
}
== Appendix 1 - Code <app-1>
Add your code here.

== Appendix 2 - Tables <app-2>
Add your tables here, see https://typst.app/docs/reference/model/table/
for reference.