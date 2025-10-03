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
// Table across pages
#show figure: set block(breakable: true)
// Table Styling
#show table.cell.where(y: 0): strong

#set table(
  stroke: (x, y) => (
    x: none,
    y: 0.2pt,
    left: if x == 0 {0.2pt} else {0pt},
    right: 0.2pt
  ), 
  align: (x, y) => (
    if x > 0 { center }
    else { left }
  ),
  fill: (_, y) => if y == 0 {rgb("#4173c0")},
)
#show table.cell: it => {
  if it.y == 0 {
    set text(white)
    strong(it)
  } else {
    it
  }
}

// Code block grey background
#show raw.where(block: true): set block(fill: luma(250), inset: 1em, radius: 0.5em, width: 100%)

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
#head[Abstract]
// Harry
Electricity demand forecasting is an essential part of the modern electricity grid #highlight[REF]. In particular, it allows key stakeholders the ability to make adjustments to supply and pricing levers to ensure stable and reliable operation. \
This paper thoroughly investigates and compares the performance of four distinct methodologies for the task of forecasting the net total electricity demand over a 24 hour period. On the 2019 test set, the Linear Regression, XGBoost, LSTM, and Transformer acheived MAPES of 2.22%, #highlight[XXX%], 2.76%, and 2.79% respectively.
These results show that basic methods are able to produce results that are competitive with more complex models, and stakeholders should not disregard them in their design of forecasting algorithms. 

#pagebreak()
#outline(
  title: [#head[Contents]], 
  target: heading.where(level: 1).or(heading.where(level: 2)).or(heading.where(level: 2))
)

#pagebreak()
#counter(page).update(1)
#set page(numbering: "1", margin: 40pt)
= Introduction
== Problem & Motivation 
Electricity demand forecasting has a central role in maintaining the stability and efficiency of modern power systems. Short-term demand forecasting is essential to estimate the daily need for electricity @nw13. Accurate short-term predictions - especially up to 24 hours - helps bodies such as regulators, businesses and electricity operators to make informed decisions to better balance supply and demand, improve infrastructure, avoid costly over- or under-supply situations, aid policy and regulation making, optimise strategies for traders and more. 

Climate factors, namely temperature, weather and unexpected events, are known to strongly influence electricity consumption. By using historical electricity demand data and analysing climate factors, forecasting electricity demand using various data modelling techniques has been proved to produce accurate results in literature. 

Endgame Economics is a technology-led consultancy that advises their clients that operate in the National Electricity Market (NEM). As such, an important question they wish to address is: How can electricity demand accurately be forecasted, specifically for the New South Wales, Australia Region?

== Research Question 
The goal of this project is to address Endgame Economics’ question, which can be explored through the research question: “How can different data modelling techniques be used to accurately forecast short-term (24-hour ahead) electricity demand in NSW with temperature data and other factors?“ 
== Proposed Project Solution & Design Choices 

This research question can be guided by the following: 

#underline[1. What statistical and machine learning techniques can be developed for this forecasting task?]

Prior studies show that statistical and machine learning methods are effective for short term load forecasting. This includes the application of linear regression [ADD REFERENCES], tree-based ensemble methods such as gradient boosting and gradient bagging implementations [ADD REFERENCES], Long Short-Term Memory models [ADD REFERENCES] and Transformer models [ADD REFERENCES]. 

#underline[2. What level of accuracy can these models achieve when using standard evaluation metrics such as Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Root Mean Squared Error (RMSE)? ]

The literature demonstrates accuracy comparison using these metrics for Australian and international datasets @ref1 @ref2 @ref7. AEMO highlights the importance of forecast precision in operations @ref5 @ref6. 

#underline[3. Does adding multiple explanatory inputs (e.g. min/max/avg temperature @ref12 @ref18, heating & cooling degree days @ref2 @ref13 @ref18, day-of-week @ref12 @ref13 @ref15, seasonality @ref13 @ref15) improve accuracy over only temperature?]

Evidence across the studies referenced shows that incorporating such environmental and calendar features significantly enhances predictive performance, with systematic reviews further confirming the value of including external variables @ref7.

#underline[4. How do the top-performance models in this project compare to broader literature on electricity demand forecasting? ]
Benchmarking against published results in Australia @ref1 @ref2 @ref3 and internationally @ref4 @ref7 is important. 

By answering these questions, this research seeks to provide both practical insights and results for Endgame Economics and their clients. The findings will help identify the best methods for forecasting demand in NSW, and contribute to the growing literature on combining data modelling techniques in energy forecasting, particularly for applications in Australia. 


#pagebreak()
= Literature Review
// All of the following is from the project plan
== Importance of Forecasting Electricity Demand
A common through line of the recent studies into modelling energy demand forecasts is the clear need for accurate forecasting capabilities for the energy sector.  Globally, this would allow energy providers to strengthen their networks to avoid undersupply and wastage. With an adaptable and robust model that captures accurate energy requirements, governments and energy providers can make informed decisions on investment, policy and supply @ref11.  
== Forecasting Features <features>
=== Temperature
While recorded temperature is the most widely used predictor in the reviewed literature @ref8 @ref9 @ref10 many other temperature-related factors have been utilised to improve prediction accuracy.  For instance, heating and cooling degree measurements can be considered, which compares temperature measurements to a base temperature where no heating (or cooling) is necessary @ref2 @ref13.  Additionally, introducing temperature forecasts can increase prediction accuracy @ref11 @ref12) Restricted minimum and maximum temperature forecasts can also be used in models @ref12. 
=== Day of Week
As energy demand is not consistent daily, many studies have explored strategies to introduce factors to mitigate this variability.  Some of these include grouping working days together and non-working days together such as weekends @ref2 @ref13, region-specific public holidays @ref9 @ref11 and school holidays @ref14. 
=== Monthly and Seasonality Trends
Multiple strategies have also been utilised to account for variability in temperatures and energy demand based on the time of the year. These have differed depending on the format of the research and the methodologies used.  One of the widest used features is to encode the month of the year as a categorical predictor @ref8 @ref12 @ref13, and an alternate feature is to aggregate months into climate-based seasons @ref2 @ref15. 
=== Lag Demand Features
Lagged demand features provide information on the persistence of electricity use across time. Prior day’s maximum and minimum demand are discussed as strong predictors for next-day demand in short-term load forecasting @ref1,@ref2. Studies also employ lagged  demand at different horizons such as daily, weekly, and monthly intervals to represent temporal dependence @ref3. The interaction of lagged demand with climatic variable and calendar dummies has been shown to improve forecasting accuracy @ref4. Some regression models also incorporate lagged demand with additive or semi-parametric specifications which estimate nonlinear effects while maintaining model interpretability @ref2,@ref5.
=== Precipitation
Precipitation has been used as an additional climatic feature in electricity demand forecasting, though its influence is often weaker compared to temperature or humidity. Several studies have used rainfall and rainy days into regression and hybrid models to explain demand variations in regions with seasonal rainfalls @ref6,@ref6. In humid or monsoon climates, rainfall is associated with cooling and heating requirements, and may indirectly influence demand by altering temperature, solar radiation, and humidity levels @ref8 . 
=== Sunlight exposure
Sunlight exposure has been considered as an explanatory factor in electricity demand forecasting due to its impact on lighting, cooling requirements, and solar generation. High levels of solar exposure reduce lighting demand but may increase cooling load in warmer regions. To improve short-term forecasting accuracy, regression-based models often include solar radiation or sunshine hours as predictors, sometimes in interaction with temperature @ref8@ref9.
== Application of Data Modelling
Both traditional statistical modelling techniques and more modern machine learning algorithms were documented in the literature reviewed. 
=== Statistical Methods
Traditional statistical methods are often used for time-series based regression tasks such as forecasting energy demand.  Studies referenced the use of Multiple Linear Regression as the basic building blocks for more complex regression models such as ARIMA @ref8, Time Series Decomposition @ref11 and the novel Moving Window Regression method, which fits models over rolling time windows to adapt to changing demand patterns @ref13. 
==== Linear Regression
Linear regression is one of the most established approaches for electricity demand forecasting. It models the linear relationship between electricity demand and explanatory features such as temperature, lagged demand, and calendar features @ref10,@ref6. Extending the OLS estimation method, generalized least squares (GLS) has been used for auto-correlated errors [53] [54]. Moreover, regularized techniques such as Lasso and Ridge, have been used to improve parameter stability in high-dimensional @ref13. Empirical studies show that regression models with weather variables and day-type effects can achieve forecasting errors close to machine learning models for day-ahead and monthly forecasting horizons@ref13,@ref14,@ref15 . Due to its interpretability and consistent performance, linear regression models are often used as baseline benchmarks in electricity demand forecasting research.

=== Machine Learning Methods
The literature presented numerous machine learning methods as suitable for the task.  These included algorithms such as K-Nearest Neighbours @ref12, Random Forests @ref11, Gradient Boosting methods including XGBoost @ref15, and SVM (Support Vector Machines)@ref15. More recently, neural network-based analysis has taken place, including basic Artificial Neural Networks (ANNs), Convolutional Neural Networks (CNNs) @ref2, Long Short-Term Memory (LSTM) @ref2 and Transformer-Based Models @ref8, along with many combinations and hybrids of these methods. All were presented in the studies as having relevance based upon the dataset used and the specific research. 
==== Tree-Based Ensemble Methods
==== LSTM <lstm_app>

As described in @lstm_desc, the long short-term memory method has gained in popularity over recent times for time series prediction tasks.  The reviewed literature presents the base single layer LSTM into a densely connected layer as the most often used @lstmlit1 @lstmlit3 @lstmlit4 @lstmlit6 @lstmlit7, but also presents extensions on the base.  This includes integration with a multi-layer perceptron @lstmlit3, a stacked bi-directional LSTM @lstmlit5, a single bi-directional LSTM @lstmlit8, a CNN-LSTM hybrid @lstmlit8 and an LSTM combined with an attention mechanism @lstmlit8.  Results of these studies in the literature are summarised in @lstmlitrevtable and will be discussed briefly in @lstmlitrev_disc.

==== Transformers
The transformer was introduced in 2017 by Vaswani et al in the paper "Attention is All You Need" @google1. As the name implies, the architecture eschewed recurrence in favour of the attention mechanism that was gaining popularity at the time @transformer1. The reviewed literature shows that while it excels in natural language processing, it also provides competitive performance in forecasting tasks @transformer2 @transformer3 @transformer4. Modifications for this task include division of known and predicted sequences @transformer2, and integration of densely connected layers for advanced feature recognition @transformer4. The results of these studies are shown in @translitrevtable and discussed in @translitrevdisc.

== Data Modelling Results
The most common evaluation metrics used in the experiments from the sources cited in the reviewed literature were Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).  RMSE is sensitive to outliers, so its use is restricted to datasets that are missing outliers or have undertaken pre-processing to deal with these outliers.  MAE reports the average error amount (Watts) between the prediction and actual demand values.  As the studies were based on differing datasets, comparing MAE values here is unhelpful.  Instead, using MAPE as an evaluation method shows a percentage error and is thus suitable for comparison of models presented in the literature.  With a large variety of methods being utilised to model the forecast electricity demand, there was not a high variation in reported MAPE, with the highest at 5.18% (Bayesian Neural Networks) @ref9 and the lowest at 1.82% (Ensemble KNN Method) @ref12.

Based on the initial literature review (for details, see @litrevtable) four modelling methods were chosen for use in this report, and specific literature reviews were undertaken.  Information on these are outlined in the following section.   

== Model Specific
=== Linear Regression

==== Comparison Table

#figure(
  {show table.cell: set text(size: 6pt)
  table(
  columns: (auto, auto, auto, auto, auto, 120pt, 100pt),
  table.header([Study], [Dataset Location], [Dataset], [Forecast term], [Features], [Models], [Results]),
  
  [@Reg1],
  [New South Wales, Australia],
  [AEMO, Bureau of Meteorology],
  [Monthly],
  [CDD, HDD, Humidity, Rainy Days],
  [Multiple Linear Regression with VIF analysis and Backward Elimination],
  [MAPE \ MLR: 1.35%],
  
  [@reg2],
  [Thailand],
  [National electricity demand and weather datasets],
  [Half-hourly],
  [Types of days, Temperature, Historical demand, Interaction terms],
  [OLD Regression, GLS Regression, Feed Forward ANN (FF-ANN)],
  [MAPE \ FF-ANN: 2.72% \ GLS: 1.88% \ OLS: 1.97%],

  [@reg3],
  [Hokkaido, Japan],
  [Hokkaido Electric Power Company (HEPCO), Japan Meteorological Agency (JMA)],
  [Hourly],
  [Rainfall, Relative Humidity, Wind Speed, Solar Radiation, Cloud Cover],
  [Multiple linear models; Model A (deterministic terms, historical demand), Model B (Model A + Temperature), Model C (Model B + atmospheric features) using ARMAX and Bayesian estimation with Gibbs sampling],
  [MAPE \ Model A: 2.43% \ Model B: 1.98% \ Model C: 1.72%],

  [@reg4],
  [Muzaffarabad, Pakistan],
  [Water and power development authority (WAPDA), Pakistan meteorological department (PMD)],
  [Hourly],
  [Temperature, due drop, humidity, cloud cover, rainfall, windspeed],
  [MLR, KNN, SVR-linear, SVR-radial, SVR-polynomial, Random Forest, AdaBoost],
  [MAPE \ MLR: 9.23% \ KNN: 7.55% \ SVRL: 17.46% \ SVRR: 7.84% \ SVRP: 8.99 \ RF: 7.56% \ AdaBoost: 10.51%],
  
  [@reg5],
  [Tokyo, Japan],
  [Tokyo Electric Power Company Holdings],
  [Monthly],
  [Temperature],
  [MLR with non-negative least squares (NNLS) and LSE estimations, compared with SVM, RF, Lasso, LGBM, LSTM],
  [MAPE \ NNLS: 3.2% \ LSE: 3.2% \ SVM: 2.9% \ RF: 3.4% \ Lasso: 3.9% \ LGBM: 3.4% \  LSTM: 6.2%],
  
  [@reg6],
  [Thailand],
  [EGAT],
  [Half-hourly],
  [Temperature, holiday, weekday/weekend],
  [Voting regression (VR), Gradient-descent LR, OLD and generalised least-squares auto-regression (GLSAR) models, Compared with decision trees (DT) and random forests (RF)],
  [MAPE \ VR: 1.8% \ LR: 1.9% \ OLS: 1.9% \ GLSAR: 1.96% \ DT: 3.6% \ RF: 3.0%],
  
  [@reg7],
  [South Africa],
  [Eskom, South Africa's power utility company],
  [Hourly],
  [Day types, Months, Holidays, Temperature, Lagged demand at lags 1, 12, and 24],
  [Additive quantile regression (AQR), AQR with interactions (AQRI), Generalised additive model (GAM), GAM with interactions (GAMI)],
  [RMSE, MAE, MAPE \ AQR: 736.2, 568.7, 2.15% \ AQRI: 662.4, 516.5, 2.04% \ GAM: 731.5, 549.5, 2.04% \ GAMI: 648.8, 499.7, 1.86%],
  
  )},
  caption: [Regression Methods Literature Review]
) <reg-comparison>

==== Key Findings
Regression models presented strong performance in electricity demand forecasting across different regions. In New South Wales, MLR with VIF and backward elimination achieved high accuracy (MAPE 1.35%) which implies the importance of multicollinearity control and variable selection. In Thailand, GLS regression (MAPE 1.88%) outperformed OLS and ANN in the case when linear trend dominates. For Hokkaido, extending ARMAX models with Bayesian estimation and additional weather variables improved accuracy (MAPE reduced from 2.43% to 1.72%).
Machine learning models as alternatives often resulted in marginal gains, but notably regression models were consistently competitive. In Tokyo, NNLS and LSE regression matched or outperformed advanced methods such as RF and LSTM. In Muzaffarabad, regression provided good forecasts compared to SVR and KNN. In South Africa, AQR and GAM with interactions reduced errors. It shows that extending regression frameworks with interaction terms can improve predictive performance.

=== Tree-Based Ensemble Methods <tree-ensemble-section>
==== Comparison Table
#figure(
  {show table.cell: set text(size: 6pt)
  table(
  columns: (auto, auto, auto, auto, 100pt, 75pt, 170pt),
  align: horizon,
  table.header(
    [Study],
    [Dataset Location],
    [Dataset],
    [Forecast term],
    [Features],
    [Models],
    [Results],
  ),
  // --- NW12 ---
  [@nw12],[Not specified], [Kaggle],[Daily],[Historical daily average load, weather, time, day-of-week],[Random Forest Regression],[MSE: 1.897728; R²: 0.431885],

  // --- NW13 ---
  [@nw13],
  [Kerala, India],
  [Kerala State Electricity Board (KSEB)],
  [Daily],
  [Date, Demand (MW), Temperature, Dew Point, Humidity, Pressure, Wind speed, Population Density],
  [LR, DT, RF, SVR, KNN, XGBoost, ANN],
  [ Accuracy %, MAE, MSE, RMSE \

    LR: 79.39, 0.045, 0.003, 0.060 \
    DT: 70.74, 0.049, 0.005, 0.073 \
    RF: 82.72, 0.038, 0.002, 0.054 \
    SVR: 79.37, 0.048, 0.003, 0.061 \
    KNN: 79.72, 0.043, 0.003, 0.062 \
    XGB: 82.11, 0.038, 0.002, 0.054 \
    ANN: 80.31, 0.043, 0.003, 0.058
  ],

  // --- NW14 ---
  [@nw14],
  [Jordan],
  [Jordanian EMRC & DOS],
  [Daily morning & evening peak],
  [Temp (morning & evening), year-to-year load growth, daily usage (weekday/holiday), population, weekly peaks],
  [RF, XGB, CatBoost, SVR, GBM, GPR, STL],
  [
    Test MAE (MW), test MAPE (%), morning & afternoon demand \
    
    GPR: Morning 4.71/0.20, Evening 4.31/0.15 \
    GBM: Morning 31.98/1.42, Evening 25.03/0.91 \
    CatBoost: Morning 41.20/1.78, Evening 37.83/1.38 \
    XGB: Morning 38/1.68, Evening 26.78/0.97 \
    RF: Morning 33.22/1.47, Evening 27.43/1.00 \
    SVR: Morning 48.23/2.07, Evening 28.79/1.05 \
    STL: Morning 161.45/6.63, Evening 124.69/4.51
  ],

  // --- NW17 ---
  [@nw17],
  [VA, NC, SC, USA],
  [Dominion Energy / PJM (Kaggle)],
  [Two-week + daily hourly],
  [Hourly demand, time features (day, month, year, hour)],
  [RNN, LSTM, XGBoost],
  [
    MAE, RMSE, R² Score \ 
    
    Deep RNN: 0.0425, 0.02446, 0.959 \
    Stacked LSTM: 0.0329, 0.01996, 0.973 \
    XGB: 0.1321, 0.08595, 0.723
  ],

  // --- NW18 ---
  [@nw18],
  [Turkey],
  [EXIST],
  [Hourly],
  [Hourly demand, temperature, month, weekday, holidays],
  [LR, RF, XGB],
  [RMSE \
  LR: 2348.35 \
  RF: 2075.32 \
  XGB: 2038.54],

  // --- NW19 ---
  [@nw19],
  [Victoria, Australia],
  [La Trobe Energy AI/Analytics Platform],
  [Hourly],
  [Timestamp, air temp, humidity, lag weather],
  [XGB, LightGBM, CatBoost, GBM],
  [
    MAE, RMSE, nRMSE, MSE \ 
    GBM: 7.499, 9.513, 0.538, 90.92 \
    XGB: 7.856, 9.906, 0.560, 98.129 \
    LightGBM: 8.032, 10.077, 0.571, 101.555
  ],

  // --- NW20 ---
  [@nw20],
  [Nashville, TN, USA],
  [City wireless sensors],
  [Hourly],
  [Demand, air pressure, temp, humidity, wind, clouds, irradiation],
  [LSTM, XGB],
  [
    R², RMSE, MAE \
    
    LSTM: 0.838, 2.0722, 1.5275 \
    XGB: 0.859, 1.8803, 1.4299
  ],
  [@ref18],[NSW, Australia],[AEMO, BOM],[Daily], [Min temperature, max temperature, Humidity, Solar radiation],[Linear Regression, Random Forest, XGBoost],[MAPE, MSE, R²: Linear: 5.38%/260792/0.65, Random Forest: 3.03%/91809/0.88, XGBoost: 2.97%/89543/0.88],
)}, caption: [Tree-Based Ensemble Methods Literature Review]
) <tree-lit-review-table>
==== Key Findings
- Add write-up of main findings

=== Long Short-Term Memory Network
==== Comparison Table

#figure(
  {show table.cell: set text(size: 6pt)
  table(
  columns: (auto, auto, auto, auto, 150pt, auto, 100pt),
  align: horizon,
  table.header([Study], [Dataset Location], [Dataset], [Forecast term], [Features],[Models],[Results]),
  [@lstmlit1],[Brescia, Italy],[University of Brescia Energy Management System],[Daily],[Historical Load Data, Change rates, Temperature, Humidity, Wind Speed, Solar Radiation, Hour, Day, Month, Holiday],[LSTM, GRU, RNN, MLP],[RMSE, MAE \ LSTM: 1.93, 1.48 \ GRU: 2.04, 1.56 \ RNN: 2.31, 1.78 \ MLP: 2.40, 1.85],
  [@lstmlit2],[Salt Lake City, Utah, USA \ Austin, Texas, USA],[Public Safety Building, Utah \ Mesowest, East Austin  RAWS Weather Station],[83 days \ One Year],[Temperature, Humidity, Hour, Day of Week, Day of Month, Month Number, Energy Load],[LSTM-MLP],[RMSE (Utah): \ Model 1: 16.9 \ Model 2: 14.1 \ RMSE (Texas): \ Model 2: 45.35],
  [@lstmlit3],[London, England],[London Smart Meters],[Daily, 4 months, 13 months],[Sum Demand, Mean Demand],[RNN, LSTM],[RMSE: \ RNN Short term: 0.02 \ RNN Mid term: 0.03 \ RNN Long term: 0.14 \ LSTM Short term: 0.02 \ LSTM Mid term: 0.03 \ LSTM Long term: 0.15 ],
  [@lstmlit5],[Scotland],[Residential Data],[Daily, Weekly],[30 min demand, hourly temp, solar irradiation, precipitation, wind speed, year, month, day, hour, day of week, holidays, time, solar angles and moon phase],[SB-LSTM],[MAE, MAPE(%) \ Day ahead: 0.411, 1.635 \ Week ahead: 0.495, 1.973],
  [@lstmlit6],[Bareilly, India],[Smart Meter Data],[3 minutes],[3-min consumption, time, avg voltage, avg current, grid frequency, hour of day, day of week, season],[LSTM],[MAE, MSE, RMSE \ 0.0013, 0.000008, 0.0028],
  [@lstmlit8],[France],[UCI Learning Repo - Single home data],[Hourly],[1 min demand, day, month, year, hour, minute, global active power, global reactive power, voltage, global intensity, sub metering measures],[LSTM, GRU, BD-LSTM, LSTM-Att, CNN-LSTM],[RMSE, MAE, MAPE \ LSTM: 0.86, 0.63, 51.45 \ GRU: 0.86, 0.63, 51.47 \ BD-LSTM: 0.85, 0.61, 50.1 \ LSTM-Att: 0.84, 0.59, 48.36 \ CNN-LSTM: 0.61, 0.35, 34.84],
  )},
  caption: [LSTM Literature Review]
) <lstmlitrevtable>
==== Key Findings <lstmlitrev_disc>

The studies involving LSTM and hybrid LSTM-based models are somewhat hard to compare to each other, as they are completed on varied datasets.  However there are some interesting patterns that emerge from the literature.  Firstly, LSTM models tend to outperform models such as GRU and RNN when the problem horizon is short @lstmlit1 @lstmlit3 @lstmlit6 @lstmlit8, but struggle with a longer horizon @lstmlit3 @lstmlit2.  Secondly, LSTMs absolutely shine with a dataset taken from short intervals.  In some cases, having 1 and 3 minute intervals with energy demand data resulted in near perfect results @lstmlit6 @lstmlit8.  Finally, various hybrid models as described in @lstm_app can greatly improve the predictive power of the model by increasing the model's ability to pick up long term patterns and account for unexpected spikes and troughs @lstmlit8 @lstmlit2 @lstmlit5.


=== Transformer

==== Comparison Table

#figure(
  {show table.cell: set text(size: 6pt)
  table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  align: horizon,
  table.header([Study], [Dataset Location], [Dataset], [Forecast term], [Features], [Models], [Results]),
  
  [@transformer2],
  [United States],
  [Kaggle],
  [12 hour, 24 hour, and 36 hour],
  [Historical Load Data, Temperature, Date],
  [Seq2Seq, Transformer],
  [MAPE \ Seq2Seq: 11.40 \ Transformer: 10.18],

  [@transformer3],
  [Cornwall UK, and Fintry UK],
  [SMART Fintry, and Cornwall Local Energy Market],
  [Synthetic Data Absences],
  [Electricity Demand Readings],
  [Mean, Linear, SoftImpute, Mice, SVM, RF, MLP, CNN-LSTM, Transformer],
  [Average RMSE \ Mean: 47.93 \ Linear: 44.47 \ SoftImpute: 41.54 \ Mice: 35.43 \ SVM: 38.81 \ RF: 31.96 \ MLP: 34.32 \ CNN-LSTM: 28.91 \ Transformer: 20.98],

  [@transformer4],
  [Multiple unidentified cities and industries],
  [Unknown],
  [Daily and Monthly],
  [Meteorological, Social (holidays, school vacations), Historical Electricity Demand],
  [Prophet, GBDT, CNN-LSTM, Transformer-F (Transformer plus fully connected NN)],
  [Month Error \ Prophet: 0.59 \ GBDT: 0.08 \ CNN-LSTM: 0.07 \ Transformer: 0.02],
  
  )},
  caption: [Transformer Literature Review]
) <translitrevtable>

==== Key Findings <translitrevdisc>
- Add write-up of main findings

== Literature Review Key Findings
=== Different datasets will generate different results
//Add more info, include amount of data, location of dataset
//Include ideally our results should be compared to those in literature that use same dataset




=== Differences in models will generate different results



=== Different features will generate different results
//Add more info, include different features in literature a bit

== Project Novelty

This project differs from those in the reviewed literature in a number of key ways.  Firstly, the studies cited that focus on the NSW region tend to have a singular focus in terms of the methods used @ref2 @ref13 @ref14 @ref18, however we have chosen to extend the modelling to a wide range of methods to compare on the same dataset.  Our project is also unique among those focusing on the NSW region, in that it is the first making use of the transformer architecture to model energy demand forecasting.

This project has also chosen a set of features (as described in @features) that has not been seen in the specific combination that we present here.

#pagebreak()
= Material and Methods
// Harry
== Software
=== Data Storage
The provided datasets were stored in a PostgreSQL database. This has several benefits over CSV. First is column type, by explicitly casting each column to a specific type we get type safety. The second is the ability and strength of SQL joins. As we had several tables that were formatted slightly differently, using SQL made the joins easy. 

=== Data Science
Each team member conducted individual model investigations. We each used Jupyter Notebooks, and some of us also used Google Colab. This methodology allowed us to easily share and reproduce results using GitHub. All code was written in Python, and packages used specified in @prog-lang-sect.
//#figure(
//  table(
//    columns: 2,
//    table.header([Model], [Assigned To]),
//    [Linear Regression], [Saba],
//    [Tree Based Ensemble Methods], [Nidhi],
//    [LSTM], [Cameron],
//    [Transformer], [Harry]
//  ),
//  caption: [Model Investigation Assignment Schedule]
//) <work-schedule>

=== Collaboration
Microsoft Teams was used for group coordination and collaboration. A group chat allowed quick questions to be asked and answered, while team channels provided a space for more formal/important communications. In addition weekly team meetings were scheduled and held, both with just the team and also with Wei Tian. 

=== Version Control
The usage of Git was an essential part of ensuring that work was tracked and attributed correctly. In addition GitHub allows cloud based backup and sync of files, which made file sharing easy, and allowed us to track each other's progress. The repository can be found at https://github.com/harrybirdnz/ZZSC9020-Group_F.

=== Programming Language <prog-lang-sect>
Python was the primary programming language chosen for our work due to its familiarity, ease of use, and robust ecosystem. In particular the extensibility through the following packages: Panda, NumPy, Matplotlib, Seaborn, scikit-learn, Optuna, XGBoost, Tensorflow and PyTorch.

// List in text instead of table to save space
//#figure(
//  table(
//    columns: 1,
//    table.header([Package]),
//    [Pandas], [NumPy], [Matplotlib], [Seaborn], [scikit-learn],
//    [Optuna], [XGBoost], [TensorFlow], [PyTorch]
//  ),
//  caption: [Python Packages Used]
//) <py-packages>

== Description of the Data
The data used consists of four datasets.
- Electricity demand data for NSW in MW per half hour from AEMO 
- Half hourly temperature (°C) recordings from Bankstown Airport 
- Area averaged daily precipitation data in mm/day for the NSW area from NASA
- Daily solar exposure data from Sydney Airport
These datasets were each filtered to the period of interest (2016-2019).
Full dataset information can be found in @app-3[Appendix].

== Data Preparation

=== Data Cleaning
==== Demand Dataset
This file contains 22 entries where datetime is null. These must be removed.
There are 11 instances where 3 records have identical datetimes - all at 3am. This is due to daylight savings, however as we are focussed on taking aggregate values over a 24 hour period, the aggregation will still work and so we can safely ignore this.

==== Temperature Dataset
There are three missing days, the 16th, 17th, and 18th of July 2016. Dealing with these gaps is model dependent. For the two sequential models, a gap in the sequence will be detrimental to model performance. For the other two models, gaps in data should not have as big of an effect on performance, and therefore the null values can be omitted. 

=== Data Pre-Processing

==== Data filtering
Data filtered for the years 2016-2019. 2020 was removed as COVID-19 caused atypical demand patterns. Four years of data history was chosen based on the tradeoff between recency and dataset size as determined in the literature review, where various studies used multiple years of data @ref9 @ref11.
=== Feature Selection & Feature Engineering
==== Data Transformation <agg>
For ease of use with each model in Python, a master data CSV file was created (@app-code-data[Appendix]). This included the features seen in @app-data-features[Appendix]. This dataset contains many additional features that have been derived from the initial dataset. Various aggregations of all values to daily (24 hour) values (sum, average, min, max) were calculated. In addition, day of week, month, season, and weekday/weekend were derived from the date and one hot encoded. The data was finally augmented with the addition of Heating Degree Days and Cooling Degree Days values @aemo1.

==== Heating Degree Days (HDD) and Cooling Degree Days (CDD)
HDD and CDD are variables that are used to measure heating and cooling requirements. This estimate is based on the difference between the air temperature and a critical temperature set by AEMO. For New South Wales, the HDD critical temperature is 17.0 degrees C and the CDD critical temperature is 19.5 degrees C @aemo1. @hdd calculates the HDD and @cdd calculates CDD.
$ "HDD" = "Max"(0, 17 - overline(T)) $ <hdd>
$ "CDD" = "Max"(0, overline(T) - 19.5 ) $ <cdd>
==== Precipitation
Daily precipitation was derived as area-averaged time series using AIRS, SSMI GPCPDAY v3.3 (mm/day) for the period 2016-01-01 to 2019-12-31. The data was extracted over the bounding box of New South Wales (141°E,37.5°S,153.6°E,28°S) from NASA Giovanni (https://giovanni.gsfc.nasa.gov/giovanni).

==== Sunlight
Daily solar exposure data was collected from the Sydney Airport weather station, provided by the Australian Bureau of Meteorology (https://www.bom.gov.au/climate/data/)

=== Scaling and Normalisation
Ensures variables have a consistent range across variables and aids training. 
Scaling and Normalisation is an important step in the data preparation process. It ensures that variables have a consistent range which aids training. As there are four different models, the exact methods used are described in the appropriate sections (Linear Regression: @lr-dataprep, Gradient Boost: @gb-dataprep, LSTM: @lstm-dataprep, Transformer: @trans-dataprep).

=== Data Training / Testing Split <train_test_split>
Years 2016, 2017, 2018 will be used for training and 2019 used for testing.

== Assumptions
1. It is assumed that the supplied data is accurate and reliable, and that the statistical properties of the data do not change significantly over time. A known event that violates this assumption is the COVID-19 pandemic, which caused significant changes to electricity demand patterns. To mitigate this, data from 2020 was excluded from our analysis.\
2. It is also assumed that the weather data from Bankstown Airport is representative of the entire NSW region. While this is not strictly true, it is a reasonable approximation given the scope of this project.\
3. It is assumed that the features selected are sufficient to capture the underlying patterns in electricity demand. While other features may also be relevant, the selected features are based on a combination of domain knowledge and literature review.\
4. It is assumed that there are no major policy or infrastructure changes that would significantly alter electricity demand patterns during the study period.\
5. It is assumed that unusual events (e.g., natural disasters, major sporting events) that could impact electricity demand are either absent or have a negligible effect on the overall patterns observed in the data.\
6. In forecasting electricity demand, our models require an input based on their architecture and training dataset. For linear models that were trained on weather data, this is required as an input. In practice, this would come as the form of a weather forecast, and would add a layer of uncertainty and unreliability to the model. However for our purposes, we are assuming that the weather forecast is 100% accurate and therefore are supplying our Linear and Gradient Boosted models with the actual weather conditions on the day in question. As the architecture of the LSTM and Transformer model is designed to be fed sequential data from previous days, this was not implemented.

== Modelling Methods
=== Linear Regression
Linear regression is one of the mostly applied statistical methods for forecasting electricity demand @ref10 @ref53. It estimates the relationship between a response and one or more explanatory variables by fitting a linear function to the observed data. The coefficients demonstrate the marginal effect of each explanatory variable on demand, while the model minimises the sum of squared errors to provide efficient estimates under the Gauss-Markov assumptions. Compared with machine learning models such as decision trees or ensemble methods, linear regression is often preferred for its interpretability.

==== Data Preparation <lr-dataprep>
Data preprocessing included creation of polynomial temperature terms e.g., average temp2, min temp2, and max temp2. It also included encoding of calendar-based categorical features; season, quarter, weekend, holidays. Interaction features were also included for conditional effects, such as summer × average temperature, Q1 × average temperature, Q3 × sunlight exposure, and Q4 × average temperature. Precipitation was tested but dropped due to statistical insignificance
==== Model Design

The regression model was specified as an ordinary least squares (OLS) multiple linear regression (MLR) to estimate the total daily electricity demand. Let $D_d$ denote the electricity demand on day d. Mathematically, the model can be expressed as seen in @lr-eqn1.
$ D_d=β_0  +∑_(i=1)^k (β_i  X_id )+ε_d $ <lr-eqn1>
In this model, $β_0$ is the intercept, $β_i$ are the regression coefficients, $X_id$ represents the explanatory variables on day d, and $ε_d$ is the random error term assumed to be independently and identically distributed with mean zero and constant variance. For regression analysis, the explanatory variables are grouped into four components,
1. Deterministic terms ($"Det"_d$) for calendar and seasonal influences on demand. It includes indicators for weekend, holiday, seasons, and quarters.
2. Temperature terms ($"Temp"_d$) to add the functional relationship between demand and weather conditions. It includes average temperature, minimum temperature, maximum temperature, and their squared terms to estimate the nonlinear effects, cooling degree days (CD), heating degree days (HD), and sunlight exposure.
3. Historical demand terms ($"DHist"_d$) to represent the short-term persistence of demand. It includes previous day’s minimum and maximum demand.
4. Interaction terms ($"ITerms"_d$) for conditional effects i.e., the influence of temperature or solar exposure may differ across seasons and quarters. It includes summer × average temperature, Q1 × average temperature, Q3 × sunlight exposure, and Q4 × average temperature
Therefore, the regression model can be expressed as follows in @lr-eqn2.
$ D_d  = "Det"_d  + "Temp"_d  + "DHist"_d  + "ITerms"_d  +ε_d $ <lr-eqn2>
Here, $ε_d$ denotes the stochastic error component. The model is estimated using ordinary least squares (OLS) to minimise the sum of squared residuals and to obtain unbiased efficient coefficient estimates under the Gauss-Markov assumptions.

Weekend and public-holiday indicators are helpful to detect behavioral differences in electricity use. Studies report the demand being usually lower on weekends and holidays compared to weekdays @ref18. This pattern has also been recognised by AEMO’s operational forecasts @ref1,@ref2. Season and quarter indicators adjust for repeated annual variation in load. Higher demand in winter and summer relates to heating and cooling needs, while spring and autumn are milder with lower demand. Australian studies have shown that seasonal dummies improve forecast accuracy when used alongside temperature variables @ref19,@ref20.
Minimum, maximum, and average temperatures are often included with polynomial terms to incorporate non-linear effects @ref21. Studies have discussed the importance of including the non-linearity of temperature respective to demand @ref51. Studies in Japan, Thailand, and Australia have used MLR with weather and calendar features to gain low forecasting errors (e.g., MAPE < 3%) @ref17,@ref16. Lagged demand and temperature-based predictors such as HDD & CDD are discussed as strong explanatory factors for short-term load forecasting @ref23,@ref24. The motivation to apply OLS MLR in this study was the inspiration from these empirical findings, as well as the need for a computationally efficient method to benchmark against more complex machine learning models.

=== Tree-Based Ensemble Methods
Tree-based machine learning models are commonly used for regression forecasting problems, as explored in @tree-lit-review-table in @tree-ensemble-section. 

Tree-based machine learning models are based on decision trees. Decision trees recursively partition data based on the value of input features, where each internal node of the tree represents a decision based on a specific feature, leading to a subsequent split eventually leading to the leaf nodes that contain a predicted numerical outcome @nw5. An example of a decision tree for a regression problem looks like @eg-decision-tree. 
#figure(
  image("media/Example Decision Tree.png", height:17%),
  caption: [Example of Decision Tree for Regression - Image from  @nw21]
) <eg-decision-tree>
Ensemble learning is commonly used with tree-based machine learning models. Ensemble learning combines a number of different models, that usually results in models with less bias and less variance @nw4. The two most popular ensemble learning methods are boosting and bagging. Boosting is where a number of models are trained sequentially, where each model learns from the previous mistakes of the previous models @nw4. Bagging is where a number of models are trained in parallel, where each model learns from a random subset of the data @nw4. The application of boosting is found in gradient boosting decision trees and bagging is found in random forests. 
==== Gradient Boosting Decision Trees

In gradient boosting decision trees, the idea is to have many weak learners that when combined create a strong learner. All trees are connected in series and each subsequent tree or weak learner tries to minimise the error of the previous tree by fitting into the residuals of the previous step @nw4. The final model aggregates the result of each step and eventually a strong learner is created. An example of this is shown in @eg-gradboost.
#figure(
  image("media/Gradient Boost Decision Tree Example.png",height:17%),
  caption: [Example of Gradient Boosting Decision Tree - Image from @nw4]
) <eg-gradboost>
The gradient boosting decision tree models will try to minimise a particular loss function. For the purposes of this project, the models will minimise the mean squared error. The mean squared error loss is calculated by the formula:

$ "Mean Squared Error" = (1 / N) * sum_(i = 1)^N (y_i - hat(y)_i)^2 $

where y is the true value and ŷ is the predicted value and N the total number of samples. All metrics are calculated after any normalisation or scaling has been reversed.

In this project, three gradient boosting decision tree models were explored: XGBoost, LightGBM and CatBoost. These models are popular methods to perform regression forecasting and achieve accurate results according to literature @nw13, @nw14, @nw17, @nw18, @nw19 & @nw20 . They are all more advanced algorithms that have been developed on the standard gradient boosting algorithm that outperform the standard algorithm.

Each of these models will be written in Python and make use of extensive Python libraries. 

There are 2 key steps the design for each method will be separated into:  1 -  Data preparation and 2 - Model design. 

===== Data Preparation <gb-dataprep>
====== Feature Engineering <gb-featureeng>
The inclusion of lag electricity demand forecasting was explored, performing experiments with and without these features. The lag features experimented with were:
- Previous day (1 day prior) - average demand, minimum demand, maximum demand and total demand. 
- Previous week (7 days prior) - average demand, minimum demand, maximum demand and total demand. 
====== Data Scaling and Normalisation
Scaling and normalising of data is not required for tree-based decision trees as they capture non-linear relationships between features and the target variable and are not sensitive to the scale of features @nw5. 

===== Model Design
====== Model 1: XGBoost <xg-design>
Extreme Gradient Boosting (or XGBoost) is an optimised implementation of gradient boosting, designed for speed and performance @nw6. XGBoost extends a traditional gradient boosting implementation by including regularisation, that helps improve model generalisation and prevents overfitting and utilising the Newton-Raphson method (a mathematical root-finding algorithm) that enables faster model convergence and more accurate model updates at each training step. 
To implement this model, the python libraries XGBoost and XGBRegressor were used.

#underline("Hyperparameters:")

There are a number of tunable hyperparameters included for the XGBoost model. The selected hyperparameters for tuning in this project were max_depth, learning_rate, subsample, colsample_bytree, reg_alpha and reg_lambda, selected based on recommendations in literature @nw2. These hyperparameters mainly help model training and prevent model overfitting. A number of values for each hyperparameter were trialled to determine the values that resulted in the highest performing model. Descriptions of these hyperparameters and the values trialled can be found in @xgboost_hyp. 

====== Model 2: CatBoost

CatBoost (or Categorical Boosting) is a gradient boosting method that performs well for categorical features. CatBoost modifies the standard gradient boosting algorithm by incorporating ordered boosting and using target statistics for categorial feature encoding @nw6. Ordered boosting is a method that helps reduce over-fitting, by building each new tree while treating each data point as if it’s not part of the training set when the prediction is being calculated. CatBoost also encodes categorical values based on a distribution of the target variable instead of using the target value itself which helps prevent overfitting @nw6. 

To implement this model, the python libraries CatBoost and CatBoostRegressor were used.

#underline[Hyperparameters:]

There are a number of tunable hyperparameters included for the CatBoost model. The selected hyperparameters for tuning in this project were iterations, depth, learning_rate, subsample, colsample_bytree, and l2_leaf_reg, selected as they are the core training parameters for the model @nw9. These hyperparameters help prevent model overfitting. A number of values for each hyperparameter were trialled to determine the values that resulted in the highest performing model. The hyperparameter descriptions and values for learning_rate, subsample and colsample_bylevel are the same as those in XGBoost found in @xgboost_hyp. The different hyperparameters for CatBoost include iterations, depth and l2_leaf_reg. Descriptions of these hyperparameters and the values trialled can be found in @catboost_hyp. 

===== Model 3: LightGBM

LightGBM is another gradient boosting method that performs well on categorical features. One of the largest points of difference of LightGBM compared to other gradient boosting methods is that it grows trees leaf-wise and selects the leaf that provides the greatest reduction in loss @nw6. Another difference is that this method relies on an efficient histogram-based method to sort feature values and locate the best split that improves both speed and memory efficiency @nw6. Lastly, it implements gradient-based one-side sampling that focuses on the most informative data samples during training and hence speeds up model training. 

To implement this model, the python libraries LightGBM and LGBMRegressor were used.

#underline[Hyperparameters:
]

There are a number of tunable hyperparameters included for the LightGBM model. The selected hyperparameters for tuning in this project were n_estimators, max_depth, learning_rate, subsample, colsample_bytree, num_leaves and min_data_in_leaf, selected as they are the major training parameters for the model @nw10. These hyperparameters mainly help model training and prevent model overfitting. A number of values for each hyperparameter were trialled to determine the values that resulted in the highest performing model. The hyperparameter descriptions and values for n_estimators, max_depth, learning_rate, subsample and colsample_bylevel are the same as those in XGBoost as found in @xgboost_hyp. The different hyperparameters for LightGBM include num_leaves and min_data_in_leaf. 
Descriptions of these hyperparameters and the values trialled can be found in @lightgbm_hyp. 

==== Gradient Bagging Decision Trees
In gradient bagging decision trees, the idea is to use collective knowledge of several decision trees to make accurate predictions @nw16 by aggregating predictions from numerous decision trees and thereby reducing variance and improving generalisation @nw13. Gradient bagging decision trees make multiple decision trees that are trained on subsets of data. Each tree is built using a random selection of features @nw13. The final prediction is obtained by averaging the outputs @nw13. An example of this is shown in @eg-gradbag.

#figure(
  image("media/GradBaggExample.png", height:25%),
  caption: [Example of Gradient Bagging Decision Tree - Image from @nw15]
) <eg-gradbag>

In this project, the gradient bagging decision tree model explored is Random Forest. Random Forest is a popular method to perform regression forecasting and achieve accurate results according to literature @nw12, @nw13, @nw14 and @nw18. 

===== Data Prepration
The data feature engineering and scaling and normalisation for Random Forest is the same as for gradient boosting decision trees, as described in Section 3.5.2.1.1. 

===== Model Design

Random Forest is a robust and versatile machine learning technique known for its ability to handle regression challenges across multiple domains @nw16. The Random Forest regression approach can handle big datasets of various data types including continuous and categorical variables, can handle non-linear correlations, and due to its ensembling and aggregation approach is robust against overfitting when making predictions. These factors make Random Forest an ideal for load forecasting @nw12. 

To implement this model, the python libraries sklearn and RandomForestRegressor were used.

#underline[Hyperparameters:]

There are a number of tunable hyperparameters included for the Random Forest model. The selected hyperparameters for tuning in this project were n_estimators, max_depth, min_samples_split, min_samples_leaf and max_features, selected based on recommendations in literature @nw22. These hyperparameters mainly help model training and prevent model overfitting. A number of values for each hyperparameter were trialled to determine the values that resulted in the highest performing model. 
The hyperparameter description and values for n_estimators is the same as those in XGBoost as defined in @xgboost_hyp. The different hyperparameters for Random Forest include max_depth, min_samples_split, min_samples_leaf and max_features.
Descriptions of these hyperparameters and the values trialled can be found in @rf_hyp. 

=== Long Short-Term Memory Network <lstm_desc>

Long Short-Term memory (or LSTM for short) was derived in 1997 by German researchers Sepp Hochreiter and Jurgen Schmidhuber @lstm2 as a way to mitigate the Vanishing Gradient Problem found in more traditional recurrent neural networks (or RNNs)@lstm1.  They also, unlike RNNs, allow for longer memory processing, which makes them better suited for time series prediction @lstm3.  As can be seen below (@lstm-arc1), the architecture of a LSTM unit includes 3 gates: a forget gate, an input gate and an output gate along with a memory cell.
#figure(
  image("media/lstm2.png", height: 30%),
  caption: [LSTM Unit Architecture from @lstm6]
) <lstm-arc1>
Traditional RNNs pass all of the previous processed input into the immediate next layer for the future prediction @lstm3.  In contrast, the presence of the three gates in the LSTM structure allow the network to decide which information is passed through the cell and which is discarded @lstm4.  The forget gate will take the current input and the input from the previous time-step and multiply them with the current weights in the system and then add bias.  Once passed through an activation with a binarisation effect, the decision is made whether or not this information is passed on @lstm4.  The input gate processes the same information as the forget gate, but does this in such a way to work out what useful information passes through to the memory cell - through multiplying the result of a sigmoid activation function and a tanh activation function to derive a new cell state @lstm1.  Finally, the output gate formulates the output from the cell which will become the input for the next LSTM cell. The cell state (C) is updated and passed through to the next unit (in the memory cell), and the output (h) becomes the input for the next LSTM unit.  As such, the memory cell is shown to carry the important information from many previous sequences. 

As can be inferred, each cell is doing a large amount of processing of the inputs, and as such, LSTM networks can be computationally expensive with large datasets @lstm5. However, the architecture allows for these models to capture long term patterns in the provided data in a much more effective manner than traditional Convolutional Neural Networks or Recurrent Neural Networks.

The LSTM network is designed to deal effectively with sequential time series data.  As such, sequences of data for a specified time period will be fed through the LSTM to output the electricity demand prediction for the following 24 hour period.  

==== Data Preparation <lstm-dataprep>

To prepare the data for input into the LSTM, the dataset must first be broken down by a processor function that will convert the data into an input matrix of shape: $ "(number of days)" * "(number of features)" $ 

===== Feature Engineering

The input will start as univariate with a default 7 days of lagged summed temperature demand, and will become multivariate by adding appropriate additional variables in a sequential and experimental manner.

Along with the variables provided in the dataset, a variable representing temperature range will be created with the following equation: $ "temp_range" = "max_temp - min_temp" $ and a $"temp"^2$ variable will also be introduced due to the non-linear patterns in the temperature features discussed in @eda_section.  

===== Data Scaling and Normalisation

The processor function also uses the MinMaxScaler functionality from the scikit-learn Python library to scale the feature and target inputs, as this is required for the LSTM @lstm7.  This also splits the data into training and test splits as described in @train_test_split.  An additional processor function will be used to tune the model with the addition of a validation set in the early experimentation phase.  The scaling process will be inversed after the model is run to convert the output to a scale consistent with the initial data for measuring the accuracy of the predictions.

==== Model Design

The base architecture for the LSTM can be seen in @lstm-arc2.  The tanh activation function is primarily used coming out of the LSTM layer into the fully connected layer as seen  in other successful studies @lstmlit9 @lstmlit6.  The linear activation into the output layer is also standard.  A stacked LSTM structure and a convolutional neural network (CNN) - LSTM hybrid layer will also be explored due to documented success in other research @ref2 @lstmlit5.  The architectures of these models are shown on the right hand side of @lstm-arc2 below.
#figure(
  grid(
    columns: 2,
    gutter: 7em,
    image("media/lstm3.png", height: 20%),
    image("media/lstm4.png", height: 25%),
  ),
  caption: [Basic and variant LSTM architectures]
) <lstm-arc2>

The LSTM functionality will be set up by using the Tensorflow Keras library due to it's simplicity of implementation.  The Adam optimiser was chosen as it is the default optimiser for deep learning models due to it's stability and robustness @ref33.

===== Hyperparameters

There are a number of tunable hyperparameters included for the LSTM model.   In the model definition, this is num_lstm_nodes, and will be extended to include dropout and recurrent_dropout in later experiments.  For the optimiser, the tunable hyperparameter is learning_rate, and when fitting the model the tunable hyperparameter is batch_size.  The number of days in the input matrix (window_size) will be adjusted experimentally, however 7 was chosen as a default as it covers a week of data which should pick up variations in demand.  Hyperparameter tuning will be performed iteratively by the Python Optuna package as described in @model-opt.  Further information regarding these hyperparameters, their operations and values can be found in @lstm_hyp.
 
===== Overfitting Protection

To protect against overfitting, the LSTM models will integrate both ModelCheckpoint and EarlyStopping from the Tensorflow library.  

ModelCheckpoint will monitor the validation loss (when that metric is used) and will only save the best model @lstm9.  In isolation, it will run the model for the specified number of epochs (100), but will save the model when the validation loss was at it's lowest point across those epochs.

EarlyStopping monitors the validation loss as well, but will stop the model run at the point where the validation loss is at it's lowest point @lstm9.  EarlyStopping has a patience hyperparameter, which protects against the model stopping at a local minimum.  The patience will be adjusted experimentally to ensure the model fit doesn't stop too early.  EarlyStopping does not have a save mechanism, so analysis would be impossible without using ModelCheckpoint in conjunction with it.

Once the modelling switches to train test splits only, the metric used will be training loss, and EarlyStopping will no longer be used as it is unreliable without a validation set @lstm9.

=== Transformer
==== Data Preparation <trans-dataprep>
- Anything specific for your model / any feature engineering you did / data scaling / normalisation, etc. -> make these subheadings under the section
===== Feature Engineering
For this model there was no additional feature engineering. 
===== Data Scaling
The features are a range of one hot encoded and continuous features. 

==== Model Design
The Transformer network architecture (@trans-arc) was introduced in 2017 by researchers at Google @google1. It was designed to replace and outperform the primarily recurrence based models used at the time, both in increased performance and reduced training cost due to parallelisation @transformer2. The architecture is a specific instance of the encoder-decoder models that had become popular in the years prior @transformer1. The primary advancement from this architecture was in the space of natural language processing (NLP), with a myriad of models being developed and becoming familiar in the mainstream such as ChatGPT @transformer1. However, this architecture can also still be applied to forecasting problems @transformer2. 
#figure(
  image("media/transformer.png", height: 25%),
  caption: [Transformer Architecture introduced by @google1]
) <trans-arc>
The novelty of this method lies in the architecture's removal of recurrence entirely, instead relying entirely on attention mechanisms. Attention excels at learning long range dependencies, which is a key challenge in many sequence transduction tasks @google1. A self attention layer connects all positions with a constant ($OO(1)$) number of operations, executed sequentially. In comparison, a recurrent layer requires $OO(n)$ sequential operations. This means that self attention layers are faster than recurrent layers whenever the sequence length $n$ is less than the model dimensionality $d$. 

The transformer takes an input, which in NLP is a sentence or phrase, that is first converted to numbers by an embedding layer before being passed to the encoder portion of the transformer @transformer2. Sequentiality is captured through positional encoding. In our task, we aim to input sequential demand and temperature data, and output a prediction for the next 24 hours of electricity demand.

==== Input
The input can be further broken down between historical data and contextual data. Historical data is the actual temperature and demand recordings. Contextual data is that which can be extracted from the date/time, such as day of the week and month of the year. 

As the transformer interprets sequential dependencies, the sequentiality needs to be encoded into the model. This is done by adding a positional encoding function to the input which embeds each datapoint with a value which indicates its order in the sequence.

==== Structure
In essence, the transformer is just another form of neural network. As our task is sequential prediction of only one value at a time, we can simplify the architecture introduced in @google1 and eliminate the need to refeed the already generated outputs back into the model. In addition PyTorch provides an implementation of the attention and feedforward mechanism outlined in @google1 called TransformerEncoderLayer @pytorch1. This allows us to create a straightforward structure as shown in @imp-trans-arc.
#figure(
  image(
    "media/Transformer Architecture.png",
    height: 25%
  ),
  caption: "Implemented Transformer Architecture"
) <imp-trans-arc>
The novelty lies in allowing the attention mechanism (need more understanding) to capture a wide timeframe in training. 

==== Output
The output of the model is a simple floating point estimation of the total demand over the next 24 hours (Note that this is equivalent to the average demand multiplied by 24).

Inverse Scaling...

==== Evaluation
To avoid the effects of variance caused by the inherent randomness present in weight initialisation and dropout, each model was trained 5 times, and the median performing model was chosen as an accurate and reproducible representation of the hyperparameters.


== Model Optimisation Techniques <model-opt>
Both the gradient boosting-based and the neural network-based machine learning models require some kind of hyperparameter tuning to be completed to result in the most optimal solution.  

While this can be done manually, this process would be labour intensive and time consuming.  A widely used alternative in much documentation is the GridSearchCV, which looks at every combination of hyperparameters and thus locates the best one @optuna3.  Although it finds the optimal solution with absolute certainty, it also takes a lot of time and computational resources.  Alternatively, RandomSearchCV selects random combinations and finds the best combination out of those attempted @optuna3.  It is less likely to find the most optimal solution, but also takes a lot less time and resources.

Recent literature has shown the rise of a Python library known as Optuna, which assists with automated hyperparameter tuning.  It uses the method of Bayesian Optimisation @optuna3 to move through sets of hyperparameters in a probabilistic way that by definition should improve each iteration, or adjust accordingly.  It is widely used in the last few years due to its platform independence, ease of integration and the ability to visualise the tuning process @optuna2.  A study by Hanifi et al @optuna1 showed Optuna to be the most efficient technique amongst three packages designed for a similar purpose.

== Model Evaluation & Analysis


=== Accuracy
Accuracy of each model was determined using mean absolute percentage error (MAPE), defined in @mape.
$ "MAPE" = 1/N sum_(i=1)^N abs(y_i - hat(y)_i)/y_i $ <mape>
As each model will be reporting on the same dataset, bothroot mean squared error (RMSE) mean absolute error (MAE) will also be reported metrics.  The equations for these can be seen in @rmse and @mae below:
$ "RMSE" = sqrt(1/N sum_(i=1)^N (y_i - hat(y)_i)^2) $<rmse>
$ "MAE" = 1/N sum_(i=1)^N abs(y_i - hat(y)_i) $<mae>

#pagebreak()
= Exploratory Data Analysis <eda_section>
// Saba
@python1 presents the time series of daily total electricity demand (MW) in NSW from 2016 to 2019. Demand is higher in summer (Dec–Feb) and mid-winter (June–August) due to air conditioning and heating needs, and lower in transitional months, consistent with previous Australian studies @ref19. System operators such as AEMO account for these seasonal and calendar effects using month and day-type indicators @ref20. 
Though the seasonal cycle remains dominant, demand series also demonstrate short-term fluctuations, caused by from weather, daily activity, and extreme events like heatwaves. No upward or downward trend is observed, indicating stable aggregate demand in NSW, where forecasting practice treats trend as weak but emphasizes seasonal and weather-related influences @ref30. Since this study also considers lagged demand features (minimum and maximum 30-minute daily demand), the time series of these variables is presented in Appendix 1. 
#figure(
  image("media/Daily Average 30 min.png",width: 85%),
  
  caption: "Daily total 30-min electricity demand (mW)(2016-2019)"
)<python1>

The time series of climate features shows strong seasonal variability essential for demand forecasting (Figure 10). Average daily temperature (°C) has a clear summer-winter cycles, implying heating and cooling needs. Rising temperatures have been linked to higher demand in NSW, especially in summer and spring @ref31. Precipitation appears irregular, with peaks on certain days, indicating its role as an external shock rather than a seasonal influence. Sunlight exposure shows a seasonal cycle, higher in summer and lower in winter, consistent with daylight variation and its influence on solar generation and cooling demand. Solar exposure and temperature extremes significantly improve demand forecasts, including for NSW @ref26. Accordingly, this study also incorporates daily minimum and maximum 30-minute temperatures, as well as cooling degree (CD) and heating degree (HD) variables, with their series shown in Appendix 1.

#figure(
  image("media/Python Code 2.png", width: 89%),
  caption: "Time series plots of climate variables (2016-2019)"
)<python2>

In terms of variation in demand based on type of days, illustration has been presented for week-wise variation in electricity demand (see Figure 11). It shows a systematic difference between weekdays and weekends. Median demand is consistently higher on weekdays, especially on Monday to Tuesday. By contrast, weekend medians are lower (~360,000 MW). This reduction points to the effect of reduced commercial and industrial activity during weekends. Across weekdays, Tuesday records the highest median demand. The distribution is relatively stable across weekdays, which shows similar variability in typical daily loads. Few outliers are also present across all weekdays, likely corresponding to extreme weather events or unusual system conditions.
Weekend demand not only has lower medians but also lower maxima compared to weekdays (e.g., ~450,000 MW on Saturday vs ~500,000 MW on Tuesday). This confirms a weekday–weekend effect in NSW demand, consistent with operational expectations where business and industrial loads dominate during weekdays, while residential usage is more prominent during weekends. Earlier NSW and Australian load-forecasting research explicitly recognizes the weekday/weekend effect, for example, Koprinska et al. (2012) @ref32 used separate weekday-based models for NSW load data. 
 

#figure(
image("media/python code 3.png", width: 70%),
  
caption: "Variation in total electricity demand_t (MW) across weekdays"
)<python3>


In term of monthly variation, demand in NSW shows systematic fluctuations i.e., seasonal cycle (refer to 12). The highest median demand occurs in winter, with June and July around ~420,00 MW recording high levels. It is consistent with heating requirements during colder conditions. Summer months also has high median demand, e.g., January and February both with more than 400,00 MW. These months showed high medians and greater variability. Maximum loads in these months exceed 500,000 MW, which aligns with extreme heat events and widespread air-conditioning use.
In contrast, demand is notably lower in the transitional months of autumn and spring. Median decline from March to April, and reached its lowest in October and November. Outliers are most frequent in February, May, December, and July, corresponding to months when weather extremes i.e., heatwaves or cold spells, disrupt typical load patterns. It confirms the importance of using seasonal effects when modelling electricity demand, a practice well established in Australian forecasting studies@ref31,@ref32


#figure(
image("media/python code 4.png", width: 80%),  
caption: "Variation in average electricity demand_t (MW) across months"
)<python4>

The quarterly boxplots confirm the relevance of seasonality for day-ahead demand forecasting in NSW (see Figure 13). Median demand is highest in Q1 (summer) and Q3 (winter), while Q4 records the lowest levels. This seasonal structure has two implications. First, shifts in load distributions across quarters show that models without seasonal controls risk misestimating demand during transitions between summer/winter peaks and spring/autumn troughs. Second, outliers such as extreme peaks above 500,000 MW in Q1 and anomalies in Q4 suggest weather shocks within seasons. Including quarter or seasonal dummy variables helps model to distinguish baseline demand levels, ensuring that, for example, a hot day in January (Q1) is not treated as equivalent to a hot day in October (Q4).

#figure(
image("media/python code 5.png", width: 60%),
caption: "Variation in total electricity demand_t (MW) across quarters of the year"
)<python5>

To assess the predictive value of lagged demand, scatter plots of previous-day minimum and maximum demand against current total demand are shown in Figure 14. Lagged features are widely used in short-term load forecasting due to temporal dependence in electricity use. Both variables show strong positive associations. However, the wider dispersion for minimum demand suggests it explains less variability, as overnight and off-peak loads are less stable predictors of daily total demand. In contrast, maximum demand shows a tighter fit which indicates greater reliability. Peak demand refers to the periods of system stress caused by climatic conditions, which often persist across consecutive days, making it a stronger predictor of current demand.

#figure(
image("media/python code 6.png", width: 75%),

caption: "Scatter plot of demand_t (MW) vs. lagged demand (MW) features"
)<python6>
Minimum, maximum, and average temperatures are relevant for demand forecasting due to their influence on electricity use in different ways. Minimum temperature is usually night-time conditions that influence heating needs. Maximum temperature is from daytime heat stress, which influences cooling demand.
Figure 15 presents non-linear relationships for temperature-based features and demand, estimated by a polynomial fit. Demand rises when days are very hot or very cold, while it stays lower at mild temperatures. Studies on Australian load forecasting @ref22,@ref31 also used quadratic terms to model this curved response. Among the predictors, average temperature is the strongest (R^2 = 0.53). Maximum and minimum temperatures explain less (R^2 = 0.31 and 0.29). This implies that in day-ahead forecasting for NSW, average temperature is more important, and minimum & maximum temperature are useful for extreme weather periods.


#figure(
image("media/python code 7.png", width: 100%),

caption: "Scatter plot of demand_t (MW) vs. temperature-based features"
)<python7>
Lastly, the scatterplots presented in Figure 16 are presented to observe the relationship between electricity demand and climate-based features. Precipitation shows weak relationships with demand. For daily total demand versus precipitation, estimated line shows almost no effect (R² ≈ 0). This may occur because rainfall does not directly influence electricity use. Sunlight exposure also shows a low effect (R^2 = 0.02). Cooling degree (CD) and heating degree (HD) are stronger predictors, as indicated by R^2 of 0.15 and 0.16, respectively. These features are linked with energy requirements for air conditioning and heating, directly relating weather extremes to electricity consumption. 

#figure(
image("media/python code 8.png", width: 90%),

caption: "Scatter plot of demand_t (MW) vs. rest of the climate features"
)<python8>


#pagebreak()
= Analysis and Results

== Model Performance
@modelcomp shows the best performing model from each of the methodologies. 
#figure(
  {show table.cell: set text(size: 10pt)
  table(
  columns: 6,
  table.header([Model], [Features], [Notes], [MAPE], [RMSE],[MAE]),
  [Linear regression], [Demand, Temperature,Calendar-based dummies], [Using lagged min & max demand, all temperature features (linear & quadratic), CD & HD, sunlight exposure, seasons, weekend, holiday, quarters; 80/20 train-test split], [2.22%],[11,446],[8,380],
  [XGBoost], [], [], [],[],[],
  [LSTM], [Demand, Temperature, Seasonality], [Using Sum Demand, All Temp Features including Temp^2, Seasons & Weekend/Weekday + 75/25 sequential data split], [2.76%],[15,622],[10,608],
  [Transformer], [All], [Using 2010-2018 as training + 90/10 sequential data split], [2.79%], [15,335], [10,776]
  )},
  caption: [Model Performance Comparison]
) <modelcomp>

=== Model Specific Results
==== Linear Regression
The regression model demonstrated reasonable performance with a testing RMSE of 11,446 MW, MAE of 8,380 MW, and MAPE of 2.22%. On the training set, the model achieved lower errors (RMSE of 9,291 MW, MAE of 6,891 MW, and MAPE of 1.80%), indicating that the fitted regression generalizes well to unseen demand data.
The comparison of predicted data and historical data for the training period from 2016 to 2018 and testing period of 2019 is depicted in Figure 9. For training data, it can be seen that the predicted values are very close to the historical data. For testing data, model performed best on stable demand days, such as mid-winter and mid-summer, where temperature and lagged demand effects dominate. For instance, test forecasts in June and July closely tracked actual consumption with low deviations. 
Notably, demand was overestimated for the end of December, possibly due to Christmas and New Year when commercial and industrial activity declined more sharply than the model’s holiday dummy could adjust for. Conversely, the model tended to under-predict demand on peak summer days (e.g., late January and early February) when extreme temperatures caused higher-than-expected load. Demand was also underestimated in the early December. One of the possible reasons can be the unusual high residential demand for cooling/air filtration due to extreme heat, poor air quality, and disruptions in energy use by severe bushfires @ref69.

#figure(
  {
    show table.cell: set text(size: 8pt)
    show emph: set align(center)
  table(
    columns: 4,
    table.header([Model Parameter / Variable], [Best Performing Value], table.cell(colspan:2, grid(
      columns: 2,
      gutter: 7pt,
      grid.cell(colspan:2,[Best Performing Model]), [Train Metrics], [Test Metrics]
    ))),
    [*Regression model*], [], [MAPE=1.80%,\ RMSE=9,291,\ MAE=6,891.52,\ R#super()[2]=0.93], [MAPE=2.22%,\ RMSE=11,446,\ MAE=8,380,\ R#super()[2]=0.91],
    [_Weekend_], [-32162.97 (686.72)$***$], [], [],
    [_Winter_], [15480.88 (1324.41)$***$], [], [],
    [_Spring_], [400.25 (1404.70)], [], [],
    [_Holiday_], [-33964.52 (1664.36)$***$], [], [],
    [_Quarter 1_], [-124685.58 (9584.99)$***$], [], [],
    [_Quarter 4_], [-70770.68 (6887.81)$***$], [], [],
    [_Min demand (t-1)_], [9.42 (1.40)$***$], [], [],
    [_Max demand (t-1)_], [6.75 (0.49)$***$], [], [],
    [_Avg. temperature_],  [-11527.84 (2020.48)$***$], [], [],
    [_$"Avg. temperature"^2$_], [314.36 (51.24)$***$], [], [],
    [_Min temperature_], [2059.25 (560.81)$***$], [], [],
    [_$"Min temperature"^2$_], [-80.41 (20.04)$***$], [], [],
    [_Max temperature_], [-3847.90 (782.60)$***$], [], [],
    [_$"Max temperature"^2$_], [63.50 (14.71)$***$], [], [],
    [_CD_], [1761.23 (713.28)$**$], [], [],
    [_HD_], [1226.53 (728.79)$*$], [], [],
    [_Sunlight exposure_], [61.80 (66.18)], [], [],
    [_Summer $*$ Avg. temp_], [-15.23 (52.33)], [], [],
    [_Q1 $*$ Avg. temp_], [6208.47 (471.05)$***$], [], [],
    [_Q3 $*$ Sunlight_], [-617.37 (81.31)$***$], [], [],
    [_Q4 $*$ Avg. temp_], [3410.89 (355.66)$***$], [], [],
    [*Notes: $*** p < 0.01, **p<0.05, *p<0.10$ *],[],[],[]
  )},
  caption: [Results of Regression]
)
#figure(
  image("media/Regression model actual vs predicts .png"),
caption: [Forecast plots on train and test demand series using regression model]
)


==== Tree-Based Ensemble Methods <gb-results>

#figure(
 image("media/Screenshot 2025-10-03 011838.png"),
  caption: [Error Results - Tree-Based Ensemble Methods]
)<gb-table-results>

#figure(
  stack(
    spacing: 5mm, // optional spacing between images
    image("media/Screenshot 2025-10-03 010119.png", width: 100%),
    image("media/Screenshot 2025-10-03 010433.png", width: 100%)
  ),
  caption: "Result Plots of Tree-Based Ensemble Methods"
)<gb-subplots>

#figure(
 image("media/Tree-Based Feature Importance.png",height:50%),
  caption: [Feature Importance - Tree-Based Ensemble Methods]
)<gb-feature-importance-results>


#figure(
 image("media/Tree-Based Feature Category.png",height:35%),
  caption: [Feature Importance by Category - Tree-Based Ensemble Methods]
)<gb-feature-category-results>


For each tree-based ensemble model, a number of experiments were conducted:

1. Different features
Three different model feature combinations were trialled - using all features without lag features, all features with previous day lag features (as described in @gb-featureeng) and all features with previous day and previous week lag features (as described in @gb-featureeng)

2. Different hyperparameter values

Each model has different hyperparameters where various values per hyperparameter were trialled. The various values trialled per hyperparameter can be found in @xgboost_hyp. The hyperparameter values associated with the best performing model variation are shown in @gb-table-results. 

Based on the results shown in @gb-table-results, the following observations can be made:

- XGBoost’s best performing model used previous day lag demand value as a feature, resulting in a MAPE of 2.08% on the test dataset. 
- Catboost’s best performing model used both previous day and week lag demand values as features, resulting in a MAPE of 1.93% on the test dataset. It is also the best performing model out of all tree-based ensemble methods trialled in this project. 
- LightGBM’s best performing model used previous day lag demand value as a feature, resulting in a MAPE of 2.03% on the test dataset. 
- Random Forest’s best performing model used previous day lag demand value as a feature, resulting in a MAPE of 2.38% on the test dataset. This was the worst performing out of the tree-based ensemble methods trialled in this project. 

Feature importance analysis was also conducted on the best performing model of each model type, as observed in @gb-feature-importance-results and @gb-feature-category-results. 
@gb-feature-category-results maps each feature into a category. The mapping of features to categories can be found in @app-5.

From these figures, the following observations can be made:

- XGBoost’s most important feature was previous day’s maximum demand. The next few most important features included lag features such as previous day’s maximum demand, previous day’s total demand and previous day’s average demand, if it’s a weekday or weekend and if it’s winter. The feature categories that highly contribute to XGBoost were day lag features and day of the week. The features least important to the model were weather features. 
- CatBoost’s most important feature was the previous day’s average demand. The next few most important features included lag feature previous day’s total demand, if it’s a weekday, cd value and temperature values average and maximum temperature. The feature categories that highly contributed to CatBoost were day lag features, weather and day of the week. The features least important to the model were month and seasonality. 
- LightGBM’s most important feature was sunlight. The next few most important features include lag features such as previous day’s maximum demand and previous day’s average demand, temperature values maximum and average temperature and precipitation. The feature categories that highly contributed to LightGBM were weather and day lag features. The features least important to the model were season, day of the week, month and temperature derived. 
- Random Forest’s most important feature was the previous day’s maximum demand. The next few most important features included lag features such as previous day’s average demand, previous day’s total demand and previous day’s minimum demand, average temperature and cd value. The feature category that highly contributed to Random Forest was day lag features. The features least important to the model were temperature derived, season and month. 

@gb-subplots shows the actual forecast demand values versus the predicted values per model. Many plotted data points are close to the perfect predicted line, with a R² score of 0.92 for CatBoost, 0.9 for XGBoost & LightGBM and 0.88 for Random Forest. This suggests all models do a good job at predicting forecast demand values and are effectively capturing the relationships and patterns in the data. Comparatively, CatBoost’s datapoints are less scattered and closer to the perfect predicted line compared to the other models. At the higher demand values over 450000 that are also more uncommon given there are not as many data points, all models seem to mostly underestimate the demand values. Comparatively, CatBoost does the best at predicting demand values over 450000 as the data points are closer to the perfect prediction line.

@gb-subplots shows the residuals plot per model. The scatter of data points seem to be fairly random across all models, with no clear pattern and residuals don’t seem to increase or decrease in variance in a particular way. These make for good residual plots and suggest all models perform well.

@gb-subplots shows the actual and predicted forecast values on a time series plot, showing all the test dataset per model. Both actual and predicted line graphs per model look fairly similar, suggesting each model performs well to predict demand values. Further, it can be observed that the models behave conservatively for when demand has high peaks, underpredicting demand values a little. For low peaks, the models seem to overpredict demand a little. These observations show that the models mainly struggle with capturing the edges of demand values but according to MAPE values, on average the models are able to predict demand within an error margin of within 1.93% to 2.38% of the actual demand values (depending on which model).  


==== Long Short-Term Memory

The model runs for LSTM were initially performed to discover the optimal feature set for accurate energy demand predictions.  As can be seen in @lstm_res_4sel below, using the 7 day lagged demand window resulted in a test MAPE under 4%.  The creation of the temp_range feature decreased the overfitting of the model. Experimentation with different window sizes decreased the effectiveness of the model, and within the constraints of the experimentation, 7 days remained the optimal window size.

Precipitation and sunlight resulted in worse metrics, so were left out of the feature set for subsequent runs, and showed that temperature was the main driver of predictive accuracy prior to the introduction of seasonality-based features.  Specific to the LSTM structure, it was evidently able to learn the same patterns from the temperature data than it could from the other weather-related features.

Experimentation with seasonality showed that the combination of seasons and weekday/weekend was optimal.  Interestingly, the models learnt nearly identical information from the day of the week features and the weekday/weekend features, so the latter was chosen due to computational considerations.

Consideration of hybrid models was undertaken, but neither the Stacked-LSTM model or the CNN-LSTM hybrid model increased the efficacy of the results.  In fact, the CNN-LSTM model performed the worst out of any experiment.  Attempting an LSTM-Attention model at the end of experimentation resulted in a good prediction, however there were signs of overfitting and the results on the test set were not optimal.

The final optimal model was reached by iteratively introducing the $"temp"^2$ feature, the dropout and recurrent_dropout hyperparameters and then optimising with the Optuna package.  This resulted in a training MAPE of 2.46% and a test MAPE of 2.76%.



#figure(
  {show table.cell: set text(size: 6pt)
  table(
    columns: 7,
    table.header([Model No.],[Train (Val) Test Split], [Features/Changes], [Train MAPE %],[(Val MAPE %)], [Test MAPE %], [Optimised\*]),
    [1],[50:25:25],[Sum Demand],[3.95],[3.79],[3.88],[No],
    [5],[50:25:25],[Add Temp Range (Max-Min), Remove Min & Max Temp],[3.84],[3.69],[3.84],[No],
    [6],[50:25:25],[All Temperature Features excl. CDD/HDD],[3.81],[3.79],[3.88],[No],
    [9],[50:25:25],[Increase Window Size to 14],[4.24],[3.91],[4.16],[No],
    [10],[50:25:25],[Window Size -> 7, Add CDD/HDD],[3.52],[3.64],[3.86],[No],
    [12],[50:25:25],[Re-add Avg Temp, Add Precipitation],[3.41],[3.49],[3.93],[No],
    [13],[50:25:25],[Remove Precipitation, Add Sunlight],[3.44],[3.74],[3.93],[No],
    [14],[50:25:25],[Re-Add Precipitation],[3.80],[3.74],[4.17],[No],
    [15],[50:25:25],[Remove Precipitation & Sunlight, Add Seasons],[3.38],[3.54],[3.70],[No],
    [16],[50:25:25],[Remove Seasons, Add Weekday/Weekend],[2.95],[3.14],[3.42],[No],
    [18],[50:25:25],[Remove Day Of Week, Add Month],[2.87],[3.78],[4.07],[No],
    [19],[50:25:25],[Remove Month, Add Seasons & Weekday/Weekend],[2.92],[3.14],[3.31],[No],
    [20],[75:25],[Switch to Train/Test],[2.54],[N/A],[2.88],[No],
    [21],[75:25],[Remove Weekday/Weekend],[3.16],[N/A],[3.58],[No],
    [22],[75:25],[Add Weekday/Weekend, Remove Seasons],[2.70],[N/A],[3.02],[No],
    [23],[75:25],[Add Seasons, Post-Optuna],[2.71],[N/A],[2.88],[Yes],
    [25],[75:25],[Adjust Batch Size to 4],[2.57],[N/A],[2.84],[Yes],
    [26],[75:25],[Stacked LSTM Model],[2.54],[N/A],[3.08],[No],
    [27],[75:25],[CNN-LSTM Model],[2.85],[N/A],[4.87],[Yes],
    [28],[75:25],[Introduce Temp^2 to LSTM Model],[2.78],[N/A],[2.91],[No],
    [29],[75:25],[Add Recurrent Dropout],[2.71],[N/A],[2.87],[No],
    [30],[75:25],[Optimise],[2.43],[N/A],[2.76],[Yes],
    [31],[75:25],[Add Attention to LSTM],[2.46],[N/A],[3.24],[No],
    table.footer([],[],[#text(weight: "extrabold", size: 6pt)[BEST PERFORMING MODEL (Test Metrics):]],[#text(weight: "bold", size: 6pt)[MAPE: 2.78%]],[#text(weight: "bold", size: 6pt)[RMSE: 15622.76]],[#text(weight: "bold", size: 6pt)[MAE: 10606.65]]),
  )},caption: [Selected Results of LSTM Modelling]
) <lstm_res_4sel>

#figure(
 image("media/lstm_final_predictions.png", height: 20%),
  caption: [Results of LSTM]
)

#figure(
 image("media/LSTM - feature importance.png", height:30%),
  caption: [Feature Importance - LSTM]
) <lstm-feat>

==== Transformer
Model runs were run on the transformer architecture in order to determine the performance. Selected results are shown in @trans-results. Result 1 shows a baseline performed solely using demand to forecast demand. The achieved test MAPE of 3.73% is a strong result. 

#figure(
  table(
    columns: 7,
    table.header([Result],[Dataset], [Train Test Split], [Features], [Train MAPE], [Test MAPE], [Optimised\*]),
    [1],[2016-2019], [75:25], [Sum Demand], [3.43%], [3.73%], [Yes],
    [2],[2016-2019], [75:25], [Optimal Features\*\*], [2.12%], [2.88%], [Yes],
    [3],[2016-2019], [75:25], [All], [1.83%], [2.85%], [Yes],
    
    [4],[2010-2019], [90:10], [Sum Demand], [3.00%], [3.71%], [No],
    [5],[2010-2019], [90:10], [All except precipitation and sunlight], [1.94%], [2.79%], [Yes],
  ),
  caption: [Selected Transformer Results]
) <trans-results>


Optuna was used for hyperparameter optimisation. A tuning run was set up to selectively trial the inclusion of features additional to the sum of demand. The results of this can be seen in @trans-feat-imp.

#figure(
 image("media/transformer feature importance.png", width:100%),
caption: [Feature Importance - Transformer]
) <trans-feat-imp>

Result 3 shows that the inclusion or exclusion of features did not have an appreciable effect on model performance.\
It was hypothesised that expansion of the dataset to the full 10 year range would produce better results. Results 4 and 5 use the same test set of 2019 but expand the training set by 5 years. As shown, results do improve. \
@trans-best-output shows the output graphs from Result 5, while @trans-best-params shows the parameters used.

#figure(
 image("media/transformer-best-output.png"),
  caption: [Transformer Result 5 Output]
) <trans-best-output>

#figure(
  table(
    columns: 2,
    inset: 8pt,
    align: left,
    table.header([Parameter], [Value]),
    [Learning Rate], [0.0001455],
    [Batch Size], [50],
    [Sequence Length], [18],
    [Encoder Layer Dimension], [64],
    [Number of Heads], [4],
    [Feedforward Dimension], [250],
    [Dropout], [0.03607],
    [Number of Layers], [1],
  ),
  caption: [Result 5 Parameters]
) <trans-best-params>
Difficulty reproducing results inspired investigation into the effects of random processes such as weight initialisation and dropout in model convergence. @trans-variance shows the resulting MAPE from 100 training iterations of a model with identical hyperparameters. 

#figure(
 image("media/transformer-model-variance.png", width: 60%),
  caption: [MAPE distribution for a single model]
) <trans-variance>
#pagebreak()
= Discussion
//
== Linear Regression
=== Best Model

The regression model is based on the insights observed from exploratory data analysis where alternative weather and seasonal features, in-addition to climate features, were analyzed against demand. It is important to mention here that precipitation (mm/day) was initially included in the regression model but was removed due to a highly non-significant coefficient. The final model specification, presented in 3.5.1.2.1 , demonstrated consistent and statistically significant effects from calendar indicators, temperature features, lagged demand features and selected interaction terms.

=== Data features
The explanatory variables incorporated deterministic terms e.g., calendar and seasonal indicators. This included dummies for weekday/weekend, holidays, seasons, and quarters. Linear and square terms of average, minimum, and maximum temperature (°C) were added to address the non-linear effects. Cooling degree days and heating degree days were used for the effect of extremes. Sunlight exposure was tested, and interaction terms were used for conditional effects between climate variables and seasons or quarters. Information on short-term persistence was included by historical demand by adding previous day minimum & maximum demand (MW).

=== Model Parameters
The coefficients confirmed the strong influence of calendar terms. Demand was significantly lower on weekends (-32,162.97, p < 0.01) and holidays (-33,964.52, p < 0.01). Winter increased demand by 15,480.88 units (p < 0.01), while spring did not show a significant effect. Quarter 1 and Quarter 4 showed large reductions in demand (-124,685.58 and -70,770.68, respectively, both p < 0.01). Lagged minimum and maximum demand were both significant predictors, with coefficients of 9.42 (p < 0.01) and 6.75 (p < 0.01), respectively.
Results of temperature features aligned with earlier observations about its non-linear effects with demand. Average temperature had a negative linear effect (−11,527.84, p < 0.01) and a positive quadratic effect (314.36, p < 0.01), consistent with increased demand under both cold and hot extremes. Minimum and maximum temperature showed similar curvature, with coefficients of 2059.25 (p < 0.01) and −3847.90 (p < 0.01) for the linear terms, and −80.41 (p < 0.01) and 63.50 (p < 0.01) for the quadratic terms, re-confirming the U-shaped demand relationship. Cooling degree days had a stronger coefficient (1761.23, p < 0.05) than heating degree days (1226.53, p < 0.10). It implied that cooling needs result in a stronger influence on demand.
Interaction terms confirmed conditional effects. Average temperature had an additional positive effect in Quarter 1 (6208.47, p < 0.01) and Quarter 4 (3410.89, p < 0.01), whereas summer showed no significant additional effect. Sunlight exposure reduced demand significantly during Quarter 3 (−617.37, p < 0.01). This is consistent with higher solar generation in that period.


== Summary of model results
Electricity demand in NSW is strongly influenced by calendar, seasonal, and weather-related factors. Demand is lower on weekends and holidays which aligns with the reduced load on non-working days. This pattern occurs due to lower industrial and commercial activity @ref54 @ref63. Winter is associated with higher demand, which points to the heating needs. Demand rises due to increased use of electric heaters and longer lighting hours during shorter days @ref64. Whereas, quarter indicators show notable reductions in demand during Q1 and Q4. It aligns with annual demand cycles i.e., lower demand due to milder temperatures and holidays in late summer–autumn and spring-early summer @ref63. 
Lagged demand features are significant, implying the temporal dependence in short-term forecasts. This pattern is consistent with findings that electricity consumption tends to follow strong autocorrelation due to recurring daily usage cycles @ref64 @ref61. Temperature effects are non-linear; negative linear and positive quadratic terms for average, min, and max temperatures. It confirms that both hot and cold extremes result in high demand. Hot weather increases cooling needs due to air conditioning use, while cold weather raises heating requirements, both contributing to peak loads @ref50 @ref49.
Interaction terms show seasonal asymmetries, such as higher sensitivity to temperature in Q1 and Q4. These quarters coincide with transitional seasons when households rely more on heating or cooling during unexpected extremes, making demand more temperature responsive. In contrast, reduced demand from sunlight in Q3 is likely due to higher solar generation offsetting household consumption @ref53,@ref61. Lastly, cooling degree days demonstrate stronger effects than heating degree days. This result points to the dominant role of cooling needs in NSW, as air conditioning demand during hot periods outweighs heating requirements in winter @ref53 @ref65.

== Tree-Based Ensemble Methods

=== Best Model

The best model out of the tree-based ensemble methods was CatBoost, producing a MAPE of 0.20% on the training dataset and a MAPE of 1.93% on the test dataset. These results are close to those found in literature for tree-based ensemble methods in @tree-lit-review-table. The two most comparative literature comparisons that predict daily forecast demand and report MAPE results (other results are scaled differently to those in this project) are @ref18, with the best performing method being XGBoost with a MAPE of 2.97% and @nw17 with CatBoost with a MAPE of 1.78% for morning demand forecasting and MAPE of 1.38% for evening demand forecasting. For studies that forecast electricity demand in Australia in @app-2, the produced CatBoost model performs similarly, where the best performing model produces a MAPE of 1.82% @ref12. 

=== Data Features

The best performing CatBoost model used all features including weather (temperature values, precipitation, sunlight), season, day of the week, month and additional temperature measures (cooling degree and heating degree), as well as both day and week lag demand features. This model heavily relied on previous day lag demand values, weather features, day of the week categorical data and mildly on cooling and heating degree measures and previous week lag demand values, as observed in @gb-feature-category-results. Monthly and seasonality features did not contribute as much to model performance. 

=== Model Parameters

The hyperparameter values that produced the best performing CatBoost model were iterations: 500, depth: 6, learning rate: 0.15, subsample: 0.7, colsample_bytree: 0.6 and l2_leaf_reg: 1.0. The higher value of iterations suggests that the model needed additional trees to learn the complex relationships in the data. The values for subsample and colsample_bytree suggest that some feature selection helps improve model results, by combining results with lower correlated data samples and features. 

=== Summary of Model Results:

Several observations can be made from the figures in @gb-results:
- Overall, the combination of multiple explanatory features such as min/max/avg temperature, heating and cooling degree days, day of week and seasonality can help accurately forecast day-ahead electricity demand with tree-based ensemble methods. This can be observed in @gb-table-results, where without lag features, the various models trialled produce results ranging from MAPE 2.49% to 2.75% on the test dataset. These results are similar (with some variation, sometimes better or not as good) to those in literature that use tree-based ensemble methods such as @ref18 and @nw17, and in comparison to other modelling methods as seen in @app-2. 
- Overall, adding some combination of lag demand values increased model performance. This is consistent across all tree-based ensemble methods, as observed in @gb-table-results, improving model performance and decreasing forecasting errors by MAPE 0.56% for CatBoost (day and week lag), MAPE 0.45% for XGBoost (day lag), MAPE 0.48% for LightGBM (day lag) and MAPE 0.37% for Random Forest (da lag). This suggests that electricity demand patterns are cyclical, where past demand values influence future ones. It also suggests that there are daily and weekly electricity demand patterns, and that with this additional context, the models are able to forecast demand more accurately. 
- Each model has different features that highly contribute to the accuracy of the model. 
  - As aforementioned, adding lag demand values helped decrease the error margin across all models. For XGBoost, CatBoost and Random Forest, the previous day lag demand features had the most feature influence over the model. 
  - For LightGBM, weather features - specifically sunlight and precipitation - showed to have the most feature influence over the model. 
- The addition of variables such as if it was a weekday or weekend, temperature values (min/max/avg) and cooling degree days were amongst the highest contributing features across all models, that can be observed in @gb-feature-importance-results. 
- Across all models, the models are not as good at predicting electricity demand for the last two months of the year (November and December) compared to other months, where the models overpredict demand. This could be due to the presence of more public holidays, the additional use of air conditioning and other factors over this time period that do not align with cyclical demand patterns and hence the model cannot predict demand as accurately. 
- Across the tree-based models, there does seem to be a degree of model overfitting to the training data despite the various parameters implemented per model to help prevent overfitting. This can be observed by a sizable difference in training dataset MAPE values and test dataset MAPE values, where the train MAPE is lower and hence suggests the model is memorising some of the train features. For example, in the best performing tree-based ensemble method - CatBoost - in this project, the train MAPE is 0.20% compared to the test MAPE of 1.93%. 

== Long Short-Term Memory
=== Best Model

The best model for the LSTM gave a training MAPE of 2.43% and a test MAPE of 2.76%. While this model gives a higher MAPE in comparison to the other models in this report, it still falls well within the 2-5% range identified in @lstmlit10 as usable in operational load forecasting, especially with a day-ahead horizon.  A detailed look at the test results show that the maximum and minimum predictions were well inside the range of that of the actual data (in the range of 2.2-3.3%), suggesting that the LSTM in this form is unable to account for unexpected extreme shifts in energy demand.

=== Model Architecture

This model used the basic one-layer LSTM model as shown in @lstm-arc2.  Experimentation with other architectures presented considerably less desirable results, which is in direct contrast to studies in literature  @lstmlit8 @lstmlit5.  This suggests specific limitations for LSTM models in the dataset used.

==== Data Features

The optimal model included all temperature related features, which allowed it to capture the most accurate information from the data.  As @lstm-feat shows, the most significant of these features were cooling degree days and the minimum and maximum temperature measurements followed closely by the transformed average temperature,  The importance of this particular feature solidifies the need to analyse patterns in the base data prior to modelling.  Outside of temperature related features, as was seen in the iterative model runs, while the sum of 30 minute demand over a rolling 24 hour period remained the primary predictor, this was closely followed by the delineation between weekends and weekdays.  This shows that the difference in demand between weekend and weekdays is required for the LSTM to learn adequate patterns.  It is important to note, especially in relation to the focus of this report, that adding relevant features to the model outside of the primary predictor decreased test MAPE by 1.12%.

==== Model Parameters

The most significant parameter for the effectiveness of this model was the 7 day rolling window.  Efforts to change this resulted in poor metrics, which shows the LSTM model needs to capture the variations over a single week to be most accurate.  The network also needed to integrate dropout mechanisms to ensure it was not overfitting on the training data, and thus was able to generalise to an unseen test dataset.

=== Summary of model results

While remaining adequate for the prediction task in this report, the LSTM model was outperformed by the regression models and tree-based ensemble methods.  The suggestion here is that the dataset is too limited for the LSTM model to perform as well as the architecture was designed to.   Studies have suggested that LSTM models require a large dataset to perform optimally @lstmlit11.

As such, further experimentation was completed, extending the dataset to 10 years of data (see @lstm_res_allfull).  Results showed barely any improvement with a larger training set.  This leads to the conclusion that the size of the data is not important, but rather that improvement could be found in further experimentation around the granularity of the data @lstmlit12, namely by reversing the aggregation of data described in @agg and working with hourly data.

//- Overall summary of model results

== Transformer
=== Feature Importance
As shown in @trans-feat-imp, temporal features—particularly day of week indicators—demonstrated the highest importance, followed by demand-related variables and temperature metrics. Notably, minimum and maximum temperature values outperformed average temperature in feature importance rankings, suggesting extreme weather conditions may have greater impact on energy demand patterns.

Despite this feature hierarchy, Results 2 and 3 revealed that model performance remained robust regardless of feature inclusion or exclusion. This resilience to input variations highlights a key strength of the transformer architecture's self-attention mechanism, which can effectively weight relevant features during processing.

=== Best Model
The best model was the result of using all features (apart from sunlight and precipitation as these were not available) on the 2010-2019 dataset, with a test MAPE of 2.79%. This is not a significant improvement on the results produced on the standard dataset.

==== Architecture
Key architectural parameters influencing performance included transformer encoder dimensionality, feedforward network dimensions, and the number of encoder layers. The optimal configuration utilised just a single transformer layer, suggesting that deeper architectures provided diminishing returns for this specific forecasting task. This simplicity contrasts with typical transformer applications but aligns with the fact that the dataset is significantly smaller than that used for natural language processing applications.

==== Training Stability Considerations
While median aggregation across five training runs successfully mitigated variance, residual performance fluctuations of approximately 0.05% MAPE persisted. This training instability, though modest, indicates sensitivity to initial conditions that could affect reproducible deployments.

=== Performance Comparison
This model performed comparably to the LSTM, however was also outperformed in by the linear regression and Tree Based ensemble methods. It is hypothesised that this is due to the difference in application of the dataset, where the high performing models were fed known information of the day in question, in particular weather data was assumed to be known a day in advance. Possibly this assumption was too strong, however it was not within the scope of this report to gather weather forecasting data. In addition, it is possible to modify the architecture of the Transformer in order to feed in known quantities such as date derived features into the prediction as performed by @transformer2. Further investigation would be necessary to determine if this is worthwhile.

=== Future Direction and Extension
==== Architectural Enhancement
Our simplified architecture sacrificed recursive prediction capabilities to reduce complexity. Restoring this functionality could significantly enhance accuracy, particularly for multi-step forecasting scenarios. For example, maintaining 30-minute intervals while predicting 48 steps ahead (24-hour horizon) could yield more valuable operational forecasts for energy grid management.
==== Model Selection Strategy
While median performance appropriately guided hyperparameter optimization during research, commercial deployment should prioritize best-performing instances. As evidenced in @trans-variance, the 0.5% MAPE range across training runs becomes highly significant given the narrow overall performance distribution (~1% total range across configurations). In addition, the usage of the median assumes that the distribution of performance is equal for all models, which is likely not the case. This would better serve production environments where peak performance is key.
==== Scope Expansion
The model should be tested on more data streams, the most applicable of which would be other states of Australia. This would assess the versatility of the architecture in adapting to different regions.



== Overall
--The inclusion of the days to be predicted weather data in LR and Tree Based could explain any performance gap between the methodologies.
A comparison of computational demand of each of the models would perhaps also yield interesting results, it is likely however that the linear regression and tree based ensemble methods would continue to outperform the LSTM and Transformer in this area.



#pagebreak()
= Conclusion and Further Issues
// Cameron
What are the main conclusions? What are your recommendations for the “client”?
What further analysis could be done in the future?

#pagebreak()
#head[References]
#{
  show heading: none
  bibliography(("references.bib", "bib.yml"), style: "elsevier-harvard", title: [References])
}
#pagebreak()
#counter(heading).update(1)
#set heading(numbering: "A.1")
#head[Appendix]
#{
  show heading: none
  heading(numbering: none)[Appendix]
}
== Appendix 1 - Code <app-1>
=== Data Preprocessing - SQL <app-code-data>
```sql
CREATE OR REPLACE VIEW days AS
SELECT
  d::date AS date_au,
  (d::timestamp AT TIME ZONE 'Australia/Sydney') AS datetime_au_start_tz 
FROM generate_series(
  '2010-01-01'::date,
  '2019-12-31'::date,
  '1 day'::interval
) AS gs(d);

CREATE OR REPLACE VIEW intervals AS
SELECT generate_series(
    '2010-01-01 00:00:00+00'::timestamptz,  
    '2019-12-31 23:30:00+00'::timestamptz,
    '30 mins'::interval
) AS interval_time_utc;

CREATE OR REPLACE VIEW demand AS
SELECT
  datetime,                     -- timestamptz as stored
  AVG(totaldemand) AS demand
FROM totaldemand_nsw
WHERE datetime IS NOT NULL
GROUP BY datetime
ORDER BY datetime;

DROP MATERIALIZED VIEW IF EXISTS processed_demand;
CREATE MATERIALIZED VIEW processed_demand AS
SELECT
  d.date_au,
  AVG(t.demand) AS avg_30_min_demand,
  MIN(t.demand) AS min_30_min_demand,
  MAX(t.demand) AS max_30_min_demand,
  SUM(t.demand) AS sum_30_min_demand,
  COUNT(t.demand) AS count_30_min_points
FROM days d
LEFT JOIN demand t
  ON (t.datetime AT TIME ZONE 'Australia/Sydney')::date = d.date_au
GROUP BY d.date_au
ORDER BY d.date_au;

CREATE OR REPLACE VIEW temp_sydney AS
SELECT datetime, temperature FROM temperature_nsw;

DROP MATERIALIZED VIEW IF EXISTS processed_temperature;
CREATE MATERIALIZED VIEW processed_temperature AS
SELECT
  d.date_au,
  AVG(t.temperature) AS avg_temp,
  MIN(t.temperature) AS min_temp,
  MAX(t.temperature) AS max_temp,
  GREATEST(17 - AVG(t.temperature), 0) AS hd_next_24h,  
  GREATEST(AVG(t.temperature) - 19.5, 0) AS cd_next_24h 
FROM days d
LEFT JOIN temp_sydney t
  ON (t.datetime AT TIME ZONE 'Australia/Sydney')::date = d.date_au
GROUP BY d.date_au
ORDER BY d.date_au;

CREATE OR REPLACE VIEW processed_precipitation AS
SELECT
  (datetime AT TIME ZONE 'Australia/Sydney')::date AS date_au,
  precipitation
FROM precipitation_nsw;

CREATE OR REPLACE VIEW processed_sunlight AS
SELECT
  (datetime AT TIME ZONE 'Australia/Sydney')::date AS date_au,
  sunlight
FROM sunlight_nsw;

CREATE OR REPLACE VIEW processed AS
SELECT
  d.date_au AS datetime_au,    
  CASE WHEN EXTRACT(MONTH FROM d.date_au) IN (12,1,2) THEN 1 ELSE 0 END AS is_summer,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) IN (3,4,5) THEN 1 ELSE 0 END AS is_autumn,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) IN (6,7,8) THEN 1 ELSE 0 END AS is_winter,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) IN (9,10,11) THEN 1 ELSE 0 END AS is_spring,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 0 THEN 1 ELSE 0 END AS is_sunday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 1 THEN 1 ELSE 0 END AS is_monday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 2 THEN 1 ELSE 0 END AS is_tuesday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 3 THEN 1 ELSE 0 END AS is_wednesday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 4 THEN 1 ELSE 0 END AS is_thursday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 5 THEN 1 ELSE 0 END AS is_friday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 6 THEN 1 ELSE 0 END AS is_saturday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) IN (0,6) THEN 1 ELSE 0 END AS is_weekend,
  CASE WHEN EXTRACT(DOW FROM d.date_au) IN (1,2,3,4,5) THEN 1 ELSE 0 END AS is_weekday,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 1 THEN 1 ELSE 0 END AS is_jan,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 2 THEN 1 ELSE 0 END AS is_feb,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 3 THEN 1 ELSE 0 END AS is_mar,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 4 THEN 1 ELSE 0 END AS is_apr,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 5 THEN 1 ELSE 0 END AS is_may,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 6 THEN 1 ELSE 0 END AS is_jun,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 7 THEN 1 ELSE 0 END AS is_jul,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 8 THEN 1 ELSE 0 END AS is_aug,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 9 THEN 1 ELSE 0 END AS is_sep,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 10 THEN 1 ELSE 0 END AS is_oct,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 11 THEN 1 ELSE 0 END AS is_nov,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 12 THEN 1 ELSE 0 END AS is_dec,

  pd.avg_30_min_demand,
  pd.min_30_min_demand,
  pd.max_30_min_demand,
  pd.sum_30_min_demand,
  pd.count_30_min_points,

  pt.avg_temp,
  pt.min_temp,
  pt.max_temp,
  pt.hd_next_24h,
  pt.cd_next_24h,

  pr.precipitation,
  ps.sunlight

FROM days d
LEFT JOIN processed_demand pd USING (date_au)
LEFT JOIN processed_temperature pt USING (date_au)
LEFT JOIN processed_precipitation pr USING (date_au)
LEFT JOIN processed_sunlight ps USING (date_au)
ORDER BY d.date_au;

SELECT d.date_au
FROM days d
LEFT JOIN temperature_nsw t
  ON (t.datetime AT TIME ZONE 'Australia/Sydney')::date = d.date_au
GROUP BY d.date_au
HAVING COUNT(t.*) = 0
ORDER BY d.date_au;


COPY (select * from processed)
TO '/import/processed.csv'
WITH (
    FORMAT CSV,
    HEADER true,
    DELIMITER ','
);
```

=== EDA Code
```python
#1. Time Series plot for response variable: Average 30 min demand (t)
demand_mean = bxp["demand"].mean()
plt.figure(figsize=(10,3))
plt.plot(bxp["date"], bxp["demand"], color="steelblue", linewidth=1)
plt.axhline(demand_mean, color="red", linestyle="--", linewidth=1, label="Mean demand")
#horizontal mean line
.ylabel("Electricity Demand (MW)", fontsize=9)
plt.xlabel("")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
```
```python
# 2. Time series plot for derived variables 
vars_to_plot = {"lag_min_demand": r"Min Demand$_{t-1}$ (MW)", "lag_max_demand": r"Max Demand$_{t-1}$ (MW)", "min_temp": "Min temperature (°C)", "max_temp": "Max temperature (°C)"}
fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(11, 10), sharex=True)
axes = axes.flatten()
for i, (var, ylabel) in enumerate(vars_to_plot.items()):
    axes[i].plot(bxp["date"], bxp[var], color="steelblue", linewidth=1, label=var)
    mean_val = bxp[var].mean()
    axes[i].axhline(mean_val, color="red", linestyle="--", linewidth=0.6)    
    axes[i].set_ylabel(ylabel, fontsize=8)
    axes[i].grid(True, linestyle="--", alpha=0.4)
    axes[i].tick_params(axis="x", labelrotation=0, labelsize=8)
    axes[i].tick_params(axis="y", labelsize=8)
axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

```
```python
# 3. Variation in average electricity demand_t
plt.figure(figsize=(7,4))
sns.boxplot(data=bxp, x="day", y="demand", order=day_order,
            palette="Set2", fliersize=3, linewidth=1, boxprops=dict(edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="black", linewidth=1.5))
plt.xlabel("")
plt.ylabel("Electricity Demand (MW)", fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

```
```python
# 4. Electricity demand by Month
plt.figure(figsize=(10,7))
sns.boxplot(data=bxp, x="month", y="demand", order=month_order,
            palette="Set2", fliersize=3, linewidth=1, boxprops=dict(edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="black", linewidth=1.5))
plt.xlabel("")
plt.ylabel("Electricity Demand (MW)", fontsize=11)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)

```
```python
# 5. Electricity demand by Quarter
bxp["quarter"] = bxp["date"].dt.quarter

plt.figure(figsize=(6,4))
sns.boxplot(data=bxp, x="quarter", y="demand", 
            palette="Set2", fliersize=3, linewidth=1, boxprops=dict(edgecolor="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            medianprops=dict(color="black", linewidth=1.5))
plt.xlabel("")
plt.ylabel("Electricity Demand (MW)", fontsize=11)
plt.xticks(ticks=[0,1,2,3], labels=["Q1", "Q2", "Q3", "Q4"], fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

```
```python
# 6.Scatter plot of demand at time t vs. Min_demand (t-1) and Max_demand (t-1)
scatter_pairs = [("lag_min_demand", r"Min Demand$_{t-1}$ (MW)"), ("lag_max_demand", r"Max Demand$_{t-1}$ (MW)")]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharey=True)
for i, (xvar, xlabel) in enumerate(scatter_pairs):
    ax = axes[i]
    # Scatter plot with regression line
    sns.regplot(x=bxp[xvar], y=bxp["demand"], scatter_kws={"s": 10, "alpha": 0.6, "color": "steelblue"},  line_kws={"color": "red", "lw": 1}, ax=ax)
    # equation equation and Rsq
    X = sm.add_constant(bxp[xvar])
    y = bxp["demand"]
    model = sm.OLS(y, X).fit()
    intercept, slope = model.params
    R2 = model.rsquared
    #for equation text
    ax.text(0.05, 0.95,  f"y = {intercept:.2f} + {slope:.2f}x\nR2 = {R2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    ax.set_xlabel(xlabel, fontsize=9) # Labels
    if i == 0:
        ax.set_ylabel(r"Demand$_{t}$ (MW)", fontsize=9)
    else:
        ax.set_ylabel("")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=8)

```
```python
# 7. Scatter plot of demand at time t vs.temperature-based variables (t)
temp_vars = [("min_temp", r"Min Temperature$_{t}$ (°C)"),
             ("max_temp", r"Max Temperature$_{t}$ (°C)"),
             ("avg_temp", r"Average Temperature$_{t}$ (°C)")]
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14, 4), sharey=True)
for i, (xvar, xlabel) in enumerate(temp_vars):
    ax = axes[i]
    sns.scatterplot(x=bxp[xvar], y=bxp["demand"], 
                    ax=ax, color="steelblue", s=10, alpha=0.6)
    # Polynomial regression fit (degree=2)
    X_poly = np.column_stack((bxp[xvar], bxp[xvar]**2))
    X_poly = sm.add_constant(X_poly)  # add intercept
    y = bxp["demand"]
    model = sm.OLS(y, X_poly).fit()
    intercept, beta1, beta2 = model.params
    R2 = model.rsquared
    # Generate smooth curve
    x_range = np.linspace(bxp[xvar].min(), bxp[xvar].max(), 200)
    y_fit = intercept + beta1 * x_range + beta2 * x_range**2
    ax.plot(x_range, y_fit, color="red", linewidth=1)
    # Equation text
    ax.text(0.05, 0.95,
            f"y = {intercept:.2f} + {beta1:.2f}x + {beta2:.2f}x²\n2 = {R2:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    ax.set_xlabel(xlabel, fontsize=9)
    if i == 0:
        ax.set_ylabel(r"Demand$_t$ (MW)", fontsize=9)
    else:
        ax.set_ylabel("")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=8)
    
```
```python
# 8. Scatter plot of demand at time t vs. rest of the climate features
scatter_pairs = [("precipitation", r"Precipitation$_{t}$ (mm/day)"), ("sunlight", r"Sunlight Exposure$_{t}$ (MJ/m²)"), 
                 ("CD", r"CD$_{t}$ (°C)"), ("HD", r"HD$_{t}$ (°C)")]
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharey=True)
axes = axes.flatten()  
for i, (xvar, xlabel) in enumerate(scatter_pairs):
    ax = axes[i]
    sns.regplot(x=bxp[xvar], y=bxp["demand"], scatter_kws={"s": 10, "alpha": 0.6, "color": "steelblue"}, line_kws={"color": "red", "lw": 1}, ax=ax)
    X = sm.add_constant(bxp[xvar])
    y = bxp["demand"]
    model = sm.OLS(y, X).fit()
    intercept, slope = model.params
    R2 = model.rsquared
    ax.text( 0.05, 0.95, f"y = {intercept:.2f} + {slope:.2f}x\nR2 = {R2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")) 
    ax.set_xlabel(xlabel, fontsize=9)
    if i == 0:
        ax.set_ylabel(r"Demand$_{t}$ (MW)", fontsize=9)
    else:
        ax.set_ylabel("")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=8)

```
#pagebreak()
== Appendix 2 - Tables <app-2>
=== Initial Literature Review

#figure(
  {show table.cell: set text(size: 6pt)
  table(
  columns: 6,
  table.header([Study], [Dataset Location], [Dataset], [Feature(s))], [Model(s)],[Results]),
  [@ref2],[NSW, Australia],[AEMO, BOM],[Temperature, Humidity, Wind Speed, CDD, HDD],[CNN-LSTM, DNN, CNN, LSTM],[CNN: 11.9% MAPE, CNN-LSTM: 1.89% MAPE, CNN-LSTM ATT: 1.83% MAPE],
  [@ref8],[Panama],[CND],[Hour of day, Day of week, Holiday, Historical Demand],[Transformer model (complex)],[17/18 39.56 MAE 51.10 RMSE, 19/20 38.76 MAE 54.86 RMSE],
  [@ref9],[Australia, NEM regions],[BOM],[Operational Demand, Ambient Temp, Temp Forecast, Solar PV, Day-type],[BNN],[Median MAPE % NSW: 5.18],
  [@ref12],[Australia, NEM regions],[Australian National Energy Market],[Daily Minimum Temperature, Daily Maximum Temperature, Day index, Day-type],[KNN, Ensembling of 3 models],[Model 1 - 5.1% MAPE, Model 2 - 4.8% MAPE, Model 3 - 3.6% MAPE, Ensemble - 1.82% MAPE],
  [@ref13],[Sydney Airport Station, Australia],[AEMO],[Balance point temperature, CDD, HDD, Working & non-working days, Season],[Regression-based moving window],[Winter week: 1.88% MAPE, Summer week: 4.26% MAPE, Year: 2.30% MAPE],
  [@ref14],[NSW, Australia],[BOM],[Time of day, Temperature],[GAM],[2.6% SDE],
  [@ref15],[Bangkok, Thailand & metro areas],[EGAT],[Temperature, Day-type, Season effects, Historical demand, Special events],[OLS regression, GLSAR, FF-ANN],[OLS: 1.97% MAPE, GLSAR: 1.88% MAPE, FF-ANN: 2.72% MAPE],
  [@ref16],[QLD, Australia],[AEMO],[Historical demand],[MARS, SVR, ARIMA],[MARS: 200.43 MAE, SVR: 162.36 MAE, ARIMA: 474.39 MAE],
  [@ref18],[NSW, Australia],[AEMO, BOM],[Min temperature, max temperature, Humidity, Solar radiation],[Linear Regression, Random Forest, XGBoost],[MAPE, MSE, R^2: Linear: 5.38%/260792/0.65, Random Forest: 3.03%/91809/0.88, XGBoost: 2.97%/89543/0.88],
  )},
  caption: [Literature Review Summary]
) <litrevtable>

=== Compiled Dataset Description <app-data-features>
#table(
    columns: 3,
    table.header([Feature], [Description], [Notes]),
    [datetime_au], [The date in yyyy-mm-dd format], [],
    [is_summer], [A boolean value], [derived from date],
    [is_autumn], [A boolean value], [derived from date],
    [is_winter], [A boolean value], [derived from date],
    [is_spring], [A boolean value], [derived from date],
    [is_sunday], [A boolean value], [derived from date],
    [is_monday], [A boolean value], [derived from date],
    [is_tuesday], [A boolean value], [derived from date],
    [is_wednesday], [A boolean value], [derived from date],
    [is_thursday], [A boolean value], [derived from date],
    [is_friday], [A boolean value], [derived from date],
    [is_saturday], [A boolean value], [derived from date],
    [is_weekend], [A boolean value], [derived from date],
    [is_weekday], [A boolean value], [derived from date],
    [is_jan], [A boolean value], [derived from date],
    [is_feb], [A boolean value], [derived from date],
    [is_mar], [A boolean value], [derived from date],
    [is_apr], [A boolean value], [derived from date],
    [is_may], [A boolean value], [derived from date],
    [is_jun], [A boolean value], [derived from date],
    [is_jul], [A boolean value], [derived from date],
    [is_aug], [A boolean value], [derived from date],
    [is_sep], [A boolean value], [derived from date],
    [is_oct], [A boolean value], [derived from date],
    [is_nov], [A boolean value], [derived from date],
    [is_dec], [A boolean value], [derived from date],
    [avg_30_min_demand], [Average 30-minute demand], [calculated from demand dataset],
    [min_30_min_demand], [Minimum 30-minute demand], [calculated from demand dataset],
    [max_30_min_demand], [Maximum 30-minute demand], [calculated from demand dataset],
    [sum_30_min_demand], [Sum of 30-minute demand], [calculated from demand dataset],
    [count_30_min_points], [Count of 30-minute data points], [calculated from demand dataset],
    [avg_temp], [Average temperature], [calculated from temperature dataset],
    [min_temp], [Minimum temperature], [calculated from temperature dataset],
    [max_temp], [Maximum temperature], [calculated from temperature dataset],
    [hd_next_24h], [Heating degree next 24 hours], [calculated from @hdd],
    [cd_next_24h], [Cooling degree next 24 hours], [calculated from @cdd],
    [precipitation], [Precipitation in mm/day], [],
    [sunlight], [Sunlight intensity/duration], [#highlight[need source]]
  )

=== LSTM Results
==== 2016-2019 Dataset

#figure(
  {show table.cell: set text(size: 6pt)
  table(
    columns: 7,
    table.header([Model No.],[Train (Val) Test Split], [Features/Changes], [Train MAPE %],[(Val MAPE %)], [Test MAPE %], [Optimised\*],
    [1],[50:25:25],[Sum Temp],[3.95],[3.79],[3.88],[No],
    [2],[50:25:25],[Split Overlap],[3.95],[3.85],[3.95],[No],
    [3],[50:25:25],[Remove Overlap, Add Avg Temp],[3.88],[3.75],[3.91],[No],
    [4],[50:25:25],[Add Min & Max Temp],[4.03],[3.93],[4.07],[No],
    [5],[50:25:25],[Add Temp Range (Max-Min), Remove Min & Max Temp],[3.84],[3.69],[3.84],[No],
    [6],[50:25:25],[All Temperature Features excl. CDD/HDD],[3.81],[3.79],[3.88],[No],
    [7],[50:25:25],[Increase Window Size to 10],[4.36],[4.01],[4.31],[No],
    [8],[50:25:25],[Decrease Window Size to 5],[4.33],[4.10],[4.22],[No],
    [9],[50:25:25],[Increase Window Size to 14],[4.24],[3.91],[4.16],[No],
    [10],[50:25:25],[Window Size -> 7, Add CDD/HDD],[3.52],[3.64],[3.86],[No],
    [11],[50:25:25],[Remove Avg Temp],[3.82],[3.78],[4.14],[No],
    [12],[50:25:25],[Re-add Avg Temp, Add Precipitation],[3.41],[3.49],[3.93],[No],
    [13],[50:25:25],[Remove Precipitation, Add Sunlight],[3.44],[3.74],[3.93],[No],
    [14],[50:25:25],[Re-Add Precipitation],[3.80],[3.74],[4.17],[No],
    [15],[50:25:25],[Remove Precipitation & Sunlight, Add Seasons],[3.38],[3.54],[3.70],[No],
    [16],[50:25:25],[Remove Seasons, Add Weekday/Weekend],[2.95],[3.14],[3.42],[No],
    [17],[50:25:25],[Remove Weekday/Weekend, Add Day Of Week],[2.66],[2.87],[3.15],[No],
    [18],[50:25:25],[Remove Day Of Week, Add Month],[2.87],[3.78],[4.07],[No],
    [19],[50:25:25],[Remove Month, Add Seasons & Weekday/Weekend],[2.92],[3.14],[3.31],[No],
    [20],[75:25],[Switch to Train/Test],[2.54],[N/A],[2.88],[No],
    [21],[75:25],[Remove Weekday/Weekend],[3.16],[N/A],[3.58],[No],
    [22],[75:25],[Add Weekday/Weekend, Remove Seasons],[2.70],[N/A],[3.02],[No],
    [23],[75:25],[Add Seasons, Post-Optuna],[2.71],[N/A],[2.88],[Yes],
    [24],[75:25],[Alt Optimisation],[2.54],[N/A],[2.95],[Yes],
    [25],[75:25],[Adjust Batch Size to 4],[2.57],[N/A],[2.84],[Yes],
    [26],[75:25],[Stacked LSTM Model],[2.54],[N/A],[3.08],[No],
    [27],[75:25],[CNN-LSTM Model],[2.85],[N/A],[4.87],[Yes],
    [28],[75:25],[Introduce Temp^2 to LSTM Model],[2.78],[N/A],[2.91],[No],
    [29],[75:25],[Add Recurrent Dropout],[2.71],[N/A],[2.87],[No],
    [30],[75:25],[Optimise],[2.43],[N/A],[2.76],[Yes],
    [31],[75:25],[Add Attention to LSTM],[2.46],[N/A],[3.24],[No],
  ))},
) <lstm_res_4full>

==== 2010-2019 Dataset
#figure(
  {show table.cell: set text(size: 6pt)
  table(
    columns: 7,
    table.header([Model No.],[Train (Val) Test Split], [Features/Changes], [Train MAPE %],[(Val MAPE %)], [Test MAPE %], [Optimised\*],
    [1],[60:20:20],[Sum Temp],[3.60],[3.98],[3.97],[No],
    [2],[60:20:20],[Split Overlap],[3.59],[3.99],[4.05],[No],
    [3],[60:20:20],[Remove Overlap, Add Avg Temp],[3.69],[4.06],[4.09],[No],
    [4],[60:20:20],[Add Min & Max Temp],[3.38],[3.93],[3.93],[No],
    [5],[60:20:20],[Add Temp Range (Max-Min), Remove Min & Max Temp],[3.66],[4.11],[4.15],[No],
    [6],[60:20:20],[All Temperature Features excl. CDD/HDD],[3.67],[4.10],[4.16],[No],
    [7],[60:20:20],[Increase Window Size to 10],[2.55],[3.27],[3.41],[No],
    [8],[60:20:20],[Decrease Window Size to 5],[4.19],[4.44],[4.43],[No],
    [9],[60:20:20],[Increase Window Size to 14],[2.63],[3.30],[3.25],[No],
    [10],[60:20:20],[Window Size -> 7, Add CDD/HDD],[3.51],[3.94],[4.05],[No],
    [11],[60:20:20],[Remove Avg Temp],[3.44],[3.85],[3.96],[No],
    [12],[60:20:20],[Re-add Avg Temp, Add Seasons],[3.20],[3.69],[3.70],[No],
    [13],[60:20:20],[Remove Seasons, Add Weekday/Weekend],[2.42],[2.90],[3.01],[No],
    [14],[60:20:20],[Remove Weekday/Weekend, Add Day Of Week],[2.45],[3.01],[3.12],[No],
    [15],[60:20:20],[Remove Day Of Week, Add Month],[3.10],[3.65],[3.87],[No],
    [16],[60:20:20],[Remove Month, Add Seasons & Weekday/Weekend],[2.35],[2.91],[2.93],[No],
    [17],[90:10],[Switch to Train/Test],[2.35],[N/A],[2.96],[No],
    [18],[90:10],[Remove Weekday/Weekend],[2.81],[N/A],[3.41],[No],
    [19],[90:10],[Add Weekday/Weekend, Remove Seasons],[2.45],[N/A],[2.94],[No],
    [20],[90:10],[Inherit Optuna Opt],[2.34],[N/A],[2.86],[No],
    [21],[90:10],[Alt Optimisation],[2.32],[N/A],[2.86],[No],
    [22],[90:10],[Adjust Batch Size to 4],[2.39],[N/A],[2.88],[No],
    [23],[90:10],[Stacked LSTM Model],[2.38],[N/A],[2.96],[No],
    [24],[90:10],[CNN-LSTM Model],[3.29],[N/A],[4.91],[No],
    [25],[90:10],[Introduce Temp^2 to LSTM Model],[2.45],[N/A],[3.01],[No],
    [26],[90:10],[Add Recurrent Dropout],[2.45],[N/A],[2.95],[No],
    [27],[90:10],[Inherit Optimise],[2.21],[N/A],[2.74],[No],
  ))},
) <lstm_res_allfull>

#pagebreak()
== Appendix 3 – Supplied Data Description <app-3>
  
#strong[Total electricity demand];, in 30 minutes increments for New South Wales. This data is sourced from the Market Management System database, which is published by the market operator from the National Electricity Market (NEM) system. The variables are:
  
- DATETIME: Date and time interval of each observation in the format (dd/mm/yyyy hh:mm)
- TOTALDEMAND: Total demand (MW)
- REGIONID: Region Identifier (i.e. NSW1)

#strong[Air temperature] in New South Wales (as measured from the Bankstown Airport weather station). This data is sourced from the Australian Data Archive for Meteorology. Note: Unlike the total demand and forecast demand, the time interval between each observation may not be constant (i.e. half-hourly data). The variables are:

- DATETIME: Date time interval of each observation (dd/mm/yyyy hh:mm)
- TEMPERATURE: Air temperature (°C)
- LOCATION: Location of a weather station (i.e., Bankstown weather station)

*PRECIPITATION*:
Time Series, Area-Averaged of precipitation daily 0.5 deg. [AIRS, SSMI GPCPDAY v3.3] mm/day over 2016-01-01 00:00:00Z - 2020-01-01 00:00:00Z, 
Region: 141E, 37.5S, 153.6E, 28S (Boundry box for NSW)
 https://giovanni.gsfc.nasa.gov/giovanni/#service=ArAvTs&starttime=2010-01-01T00:00:00Z&endt…\
*SOLAR EXPOSURE:*
Daily solar exposure data from weather station: Sydney Airport 
https://www.bom.gov.au/climate/data/

#pagebreak()
== Appendix 4 – Expanded Modelling Explanation <app-4>

=== Tree-Based Ensemble Methods

==== XGBoost Hyperparameter Information
<xgboost_hyp>

#underline[n_estimators:]

n_estimators is the number of trees used in the model @nw5. Having a larger number of trees / estimators can help improve performance by allowing the model to learn more complex relationships in the data @nw3, but having too high a value increases training and computation time and the risk of overfitting increases. Common values for this are between 100 and 1000 @nw7. In this project, the values for n_estimator trialled were: 100, 250, 500, 750 & 1000.

#underline[max_depth:]

max_depth is the maximum tree depth in the model @nw7. Large values result in more complex models, which can often lead to overfitting and small values result in simpler models, which can lead to underfitting @nw7. A common value for max_depth is 3, and increasing this number based on performance @nw7. In this project, the values for max_depth trialled were: 2, 4, 6, 8 & 10.

#underline[learning_rate:]

The learning rate is the boosting learning rate value @nw1. Learning rate controls the step size at which the model updates weights, where smaller values result in slower but more accurate updates, and larger values result in faster but less accurate updates @nw7. In this project, the values for learning rate trialled were: 0.02, 0.05, 0.1, 0.15 & 0.2.

#underline[subsample:]

subsample is the fraction of samples used for each tree @nw7. This helps reduce correlation between individual learners where their results are close, and combining results with low correlation to produce a better overall result @nw4. A smaller value results in smaller and less complex models that can prevent underfitting, whereas larger values result in larger, complex models which can lead to overfitting @nw7. Common values for subsample are typically between 0.5 and 1. In this project, the values for subsample trialled were: 0.6, 0.7, 0.8, & 0.9.

#underline[colsample_bytree:]

colsample_bytree is the subsample ratio of columns when constructing each tree @nw1. This is very similar to the subsample parameter but instead of sampling the rows or samples, the columns or features are subsampled. This parameter and its values behave similarly to the values of subsample. In this project, the values for colsample_bytree trialled were: 0.6, 0.7, 0.8, & 0.9.

#underline[reg_alpha:]

reg_alpha is the L1 regularisation term on weights @nw1. Larger alpha values help reduce overfitting by adding a penalty term - absolute value of the magnitude of the coefficient @nw8 - to the loss function @nw7, penalising large coefficients and shrinks feature coefficients to zero which helps select only important features @nw8. 

#underline[reg_lambda:]

reg_lambda is the L2 regularisation term on weights @nw1. Larger lambda values help reduce overfitting by adding a penalty term - adds the squared magnitude of the coefficient @nw8 - to the loss function @nw7, penalising large coefficients and handles multicollinearity by shrinking feature coefficients of correlated features instead of getting rid of them @nw8.

==== CatBoost Hyperparameter Information
<catboost_hyp>

#underline[Iterations]

Iterations specify the maximum number of trees @nw9, equivalent to n_estimators in XGBoost. In this project, the values for iterations trialled were: 100, 250, 500, 750 & 1000.

#underline[Depth]

Depth defines the depth of the trees @nw9, equivalent to max_depth in XGBoost. Typical values range from 4 to 10 @nw9. In this project, the values for depth trialled were: 2, 4, 6, 8 and 10. 

#underline[l2_leaf_reg]

L2_leaf_reg specifies the coefficient for L2 regularisation term on leaf values @nw9, equivalent to reg_lambda in XGBoost. This penalty discourages large weights in leaves, helping prevent overfitting @nw9. In this project, the values for l2_leaf_reg were: 1, 3 & 5.

==== LightGBM Hyperparameter Information
<lightgbm_hyp>

#underline[num_leaves]

num_leaves is the maximum number of leaves in each tree @nw10. Larger values make more complex trees and can improve accuracy but often overfitting is an issue. Lower values make for simpler trees and can underfit. Typical values for this parameter range between 20 and 50 @nw11. In this project, the values for num_leaves trialled were: 20, 30, 40 and 50. 

#underline[min_data_in_leaf]

min_data_in_leaf are the minimum number of data points allowed in a leaf @nw10. This parameter helps avoid overfitting. Typical values for this parameter range between 10 to 100 @nw10. In this project, the values for min_data_in_leaf trialled were: 10, 20, 30, 50 and 100.

==== Random Forest Hyperparameter Information
<rf_hyp>

#underline[max_depth:]

The description of max_depth is the same as those in XGBoost in Section 3.5.2.1.2.1. However, the typical values for max_depth for Random Forest are 5-20 according to literature @nw22. In this project, the values for max_depth trialled were: None (no limit), 5, 10 & 20.

#underline[min_samples_split:]

min_samples_split is the minimum number of samples needed to split a node @nw22. A lower number lets the tree split on small subsets of data, allowing it to capture fine-grained patterns @nw22, but often reduces training generalisation and is at risk of overfitting. A higher number of samples per split results in less splits, creating more conservative trees and reduces overfitting @nw22. Typical values for this are between 2 and 10 @nw22. In this project, the values for min_samples_split trialled were: 2, 5 & 10.

#underline[min_samples_leaf:]

min_samples_leaf is the minimum number of samples required in each leaf node @nw22. For example, having a minimum number of x leaves would mean a decision rule is only considered if it would have at least x samples on each side of the split @nw22. A higher number makes the tree base-learners more general and reduces overfitting, whereas a lower number can improve accuracy and variance but may be at risk of overfitting. Common values for this parameter are between 1 to 10 @nw22. In this project, the values for min_samples_leaf trialled were: 1, 2, 5 & 10. 

#underline[max_features:]

max_features controls how many features or predictor columns are considered when creating a split @nw22. By limiting the number of features per split, this tries to de-correlate trees by ensuring different trees explore different patterns @nw22. A higher value of features makes the trees more similar to one another, which can lead to overfitting whereas lower numbers promote more randomness and diversity which reduces overfitting but can also underfit if too low. Typical values for regression are: None (meaning all features), sqrt (meaning the square root of the total number of features) and log2 (meaning the base-2 value of the total number of features) @nw22. In this project, the values trialled were: None, sqrt and log2. 

=== LSTM Network

==== Hyperparameter Information <lstm_hyp>

#underline[num_lstm_nodes]

As a neural network variant, the LSTM layer has a number of hidden units (defined here by num_lstm_nodes) which act as memory cells, which each represent temporal patterns in the data @lstm9.  Traditionally the number of nodes used is a multiple of 4 which assists in computation @lstm8.  The default in initial runs is chosen as 64, but will be tested for optimisation in later runs from the subset [1,8,16,32,64,128].

#underline[dropout]

Dropout is a method that helps to reduce overfitting.  This is done by randomly setting activations between the input connections and the hidden nodes to zero during training @lstm10. The amount set is between 0 and 1, with a dropout value of 0.5 equating to 50% of the activations being set to zero.  The initial runs will be done without dropout, but  will be tested for optimisation in later runs in the range 0.0 - 0.5.

#underline[recurrent_dropout]

Recurrent dropout is similar to dropout, except it randomly drops connections within the LSTM layer between time steps @lstm10.  This can have the effect of destabilising the loss function, but assists in reducing over-memorisation @lstm10.  The initial runs will be done without recurrent dropout, but will be introduced towards the end of experimentation, and optimised with values in the range 0.0 - 0.3.

#underline[learning_rate]

Learning rate, as described in @xg-design, controls the step size at which models update weights.  Adam, by definition, gives each parameter it's own adaptive learning rate @ref33 however the global learning rate sets the overall scale of how often weights are updated @lstm11.  Initially, this will be set to 0.001 (the accepted Adam default @ref33) but will be tuned later using values between 0.01 and 0.0001.

#underline[batch_size]

Batch size in a neural network based model refers to the number of samples that are processed before the model updates the weights @lstm12.  The default in Tensorflow/Keras is 32 @lstm9, so that will be used initially.  In this project, as the training set is relatively small, only batch sizes from the subset of [1, 4, 8, 16, 32] will be tested for optimisation. Anything above 32 would be impractical for a dataset of the size in the project @lstm12.  

#underline[window_size]

In the context of an LSTM, the window size controls the size of the input matrix as described in @lstm-dataprep.  As we are working with time series data, a window size of 7 (corresponding to a week) will be used as default, as it should capture daily variations.  Experiments will be undertaken to increase and decrease this window size with values in the subset of [5, 10, 14].  For the sake of time limitations for the project, this will not be a part of the Optuna optimisation process.

#pagebreak()
== Appendix 5 - Feature Category Mapping <app-5>

Weather: avg_temp, min_temp, max_temp, precipitation, sunlight
Seasonality: is_summer, is_autumn, is_winter, is_spring
Day_of_Week: is_monday, is_tuesday, is_wednesday, is_thursday, is_friday, is_saturday, is_sunday, is_weekend, is_weekday
Monthly: is_jan, is_feb, is_mar, is_apr, is_may, is_jun, is_jul, is_aug, is_sep, is_oct, is_nov, is_dec
Day_Lag_Features: prev_day_avg_demand, prev_day_min_demand,  prev_day_max_demand, prev_day_sum_demand
Week_Lag_Features: prev_week_avg_demand, prev_week_min_demand, prev_week_max_demand, prev_week_sum_demand
Temperature_Derived: hd_next_24h, cd_next_24h