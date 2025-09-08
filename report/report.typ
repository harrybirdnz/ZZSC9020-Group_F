#set heading(numbering: "1.1")
#set page(margin: 110pt, numbering: none)
#let head(body) = {
  set align(center)
  set text(size: 14pt, weight: "regular")
  [#line(length: 100%)
  #body \
  #line(length: 100%)\ ]
}
#show heading: it => [#text(weight: "regular")[#it]]
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
#set text(region: "AU", size:10pt, font: "New Computer Modern")

#show figure.where(
  kind: table
): set figure.caption(position: top)


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


#pagebreak()
= Material and Methods
== Software
R and Python of course are great software for Data Science. Sometimes, you might
want to use bash utilities such as awk or sed.
Of course, to ensure reproducibility, you should use something like Git and
RMarkdown (or a Jupyter Notebook). Do not use Word!
== Description of the Data
How are the data stored? What are the sizes of the data files? How many files?
etc.
== Pre-processing Steps
What did you have to do to transform the data so that they become useable?
== Data Cleaning
How did you deal with missing data? etc.
== Assumptions
What assumptions are you making on the data?
== Modelling Methods

#pagebreak()
= Exploratory Data Analysis
This is where you explore your data using histograms, scatterplots, boxplots, nu-
merical summaries, etc.


#pagebreak()
= Analysis and Results
== A First Model
Having a very simple model is always good so that you can benchmark any result
you would obtain with a more elaborate model.\
For example, one can use the linear regression model
$ Y_i = beta_0 + beta_1x_(1i) + ... + beta_p x_(pi) + epsilon_i, i=1,...,n $

where it is assumed that the $epsilon_i$’s are i.i.d. N(0, 1).

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