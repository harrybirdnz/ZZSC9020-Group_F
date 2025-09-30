// Document Setup
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set page(margin: 80pt)
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
Electricity demand forecasting is an essential part of the modern electricity grid REF. In particular, it allows key stakeholders the ability to make adjustments to supply and pricing levers to ensure stable and reliable operation. \
... Knowledge gap ...\
Here we report on four distinct methodologies for the task of forecasting the net total electricity demand over a 24 hour period. \
... Results with values ...\
... Meaning of results ...

#pagebreak()
#outline(
  title: [#head[Contents]], 
  target: heading.where(level: 1).or(heading.where(level: 2)).or(heading.where(level: 2))
)

#pagebreak()
#counter(page).update(1)
#set page(numbering: "1")
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
== Forecasting Features
=== Temperature
While recorded temperature is the most widely used predictor in the reviewed literature @ref8 @ref9 @ref10 many other temperature-related factors have been utilised to improve prediction accuracy.  For instance, heating and cooling degree measurements can be considered, which compares temperature measurements to a base temperature where no heating (or cooling) is necessary @ref2 @ref13.  Additionally, introducing temperature forecasts can increase prediction accuracy @ref11 @ref12) Restricted minimum and maximum temperature forecasts can also be used in models @ref12. 
=== Day of Week
As energy demand is not consistent daily, many studies have explored strategies to introduce factors to mitigate this variability.  Some of these include grouping working days together and non-working days together such as weekends @ref2 @ref13, region-specific public holidays @ref9 @ref11 and school holidays @ref14. 
=== Monthly and Seasonality Trends
Multiple strategies have also been utilised to account for variability in temperatures and energy demand based on the time of the year. These have differed depending on the format of the research and the methodologies used.  One of the widest used features is to encode the month of the year as a categorical predictor @ref8 @ref12 @ref13, and an alternate feature is to aggregate months into climate-based seasons @ref2 @ref15. 
=== Lag Demand Features
Lagged demand features provide information on the persistence of electricity use across time. Prior day’s maximum and minimum demand are discussed as strong predictors for next-day demand in short-term load forecasting [34, 35]. Studies also employ lagged average demand at different horizons such as daily, weekly, and monthly intervals to represent temporal dependence [36]. The interaction of lagged demand with climatic variable and calendar dummies has been shown to improve forecasting accuracy [37]. Some regression models also incorporate lagged demand with additive or semi-parametric specifications which estimate nonlinear effects while maintaining model interpretability [35] [38].
=== Precipitation
Precipitation has been used as an additional climatic feature in electricity demand forecasting, though its influence is often weaker compared to temperature or humidity. Several studies have used rainfall and rainy days into regression and hybrid models to explain demand variations in regions with seasonal rainfalls [39] [40]. In humid or monsoon climates, rainfall is associated with cooling and heating requirements, and may indirectly influence demand by altering temperature, solar radiation, and humidity levels [41]. 
=== Sunlight exposure
Sunlight exposure has been considered as an explanatory factor in electricity demand forecasting due to its impact on lighting, cooling requirements, and solar generation. High levels of solar exposure reduce lighting demand but may increase cooling load in warmer regions. To improve short-term forecasting accuracy, regression-based models often include solar radiation or sunshine hours as predictors, sometimes in interaction with temperature [41][42].
== Application of Data Modelling
Both traditional statistical modelling techniques and more modern machine learning algorithms were documented in the literature reviewed. 
=== Statistical Methods
Traditional statistical methods are often used for time-series based regression tasks such as forecasting energy demand.  Studies referenced the use of Multiple Linear Regression as the basic building blocks for more complex regression models such as ARIMA @ref8, Time Series Decomposition @ref11 and the novel Moving Window Regression method, which fits models over rolling time windows to adapt to changing demand patterns @ref13. 
==== Linear Regression
Linear regression is one of the most established approaches for electricity demand forecasting. It models the linear relationship between electricity demand and explanatory features such as temperature, lagged demand, and calendar features [43] [39]. Extending the OLS estimation method, generalized least squares (GLS) has been used for auto-correlated errors [53] [54]. Moreover, regularized techniques such as Lasso and Ridge, have been used to improve parameter stability in high-dimensional [55]. Empirical studies show that regression models with weather variables and day-type effects can achieve forecasting errors close to machine learning models for day-ahead and monthly forecasting horizons [55] [56] [57]. Due to its interpretability and consistent performance, linear regression models are often used as baseline benchmarks in electricity demand forecasting research.

=== Machine Learning Methods
The literature presented numerous machine learning methods as suitable for the task.  These included algorithms such as K-Nearest Neighbours @ref12, Random Forests @ref11, Gradient Boosting methods including XGBoost @ref15, and SVM (Support Vector Machines)@ref15. More recently, neural network-based analysis has taken place, including basic Artificial Neural Networks (ANNs), Convolutional Neural Networks (CNNs) @ref2, Long Short-Term Memory (LSTM) @ref2 and Transformer-Based Models @ref8, along with many combinations and hybrids of these methods. All were presented in the studies as having relevance based upon the dataset used and the specific research. 
==== Tree-Based Ensemble Methods
==== LSTM <lstm_app>

As described in @lstm_desc, the long short-term memory method has gained in popularity over recent times for time series prediction tasks.  The reviewed literature presents the base single layer LSTM into a densely connected layer as the most often used @lstmlit1 @lstmlit3 @lstmlit4 @lstmlit6 @lstmlit7, but also presents extensions on the base.  This includes integration with a multi-layer perceptron @lstmlit3, a stacked bi-directional LSTM @lstmlit5, a single bi-directional LSTM @lstmlit8, a CNN-LSTM hybrid @lstmlit8 and an LSTM combined with an attention mechanism @lstmlit8.  Results of these studies in the literature are summarised in @lstmlitrevtable and will be discussed briefly in @lstmlitrev_disc.

==== Transformers
The transformer was introduced in 2017 by Vaswani et al in the paper "Attention is All You Need" @google1. As the name implies, the architecture eschewed recurrence in favour of the attention mechanism that was gaining popularity at the time @transformer1. The reviewed literature shows that while it excels in natural language processing, it also provides competitive performance in forecasting tasks @transformer2 @transformer3 @transformer4. Modifications for this task include division of known and predicted sequences @transformer2, and integration of densely connected layers for advanced feature recognition @transformer4. The results of these studies are shown in @translitrevtable and discussed in @translitrevdisc.

== Data Modelling Results
The most common evaluation metrics used in the experiments from the sources cited in were Mean Absolute Percentage Error (MAPE), Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).  RMSE is sensitive to outliers, so its use is restricted to datasets that are missing outliers or have undertaken pre-processing to deal with these outliers.  MAE reports the average error amount (Watts) between the prediction and actual demand values.  As the studies were based on differing datasets, comparing MAE values here is unhelpful.  Instead, using MAPE as an evaluation method shows a percentage error and is thus suitable for comparison of models presented in the literature.  With a large variety of methods being utilised to model the forecast electricity demand, there was not a high variation in reported MAPE, with the highest at 5.18% (Bayesian Neural Networks) @ref9 and the lowest at 1.82% (Ensemble KNN Method) @ref12.

=== Data Modelling Results Summary
The following is a summary of the results of the reviewed literature:
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

== Model Specific
=== Linear Regression

==== Comparison Table

#figure(
  {show table.cell: set text(size: 6pt)
  table(
  columns: (auto, auto, auto, auto, auto, auto, 20%),
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
  columns: (auto, auto, auto, auto, auto, auto, auto),
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
    MAE, RMSE, R2 Score \ 
    
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
)}, caption: [Tree-Based Ensemble Methods Literature Review]
) <tree-lit-review-table>
==== Key Findings
- Add write-up of main findings

=== Long Short-Term Memory Network
==== Comparison Table

#figure(
  {show table.cell: set text(size: 6pt)
  table(
  columns: 7,
  table.header([Study], [Dataset Location], [Dataset], [Forecast term], [Features],[Models],[Results]),
  [@lstmlit1],[Brescia, Italy],[University of Brescia Energy Management System],[Daily],[Historical Load Data, Change rates, Temperature, Humidity, Wind Speed, Solar Radiation, Hour, Day, Month, Holiday],[LSTM, GRU, RNN, MLP],[RMSE, MAE \ LSTM: 1.93, 1.48 \ GRU: 2.04, 1.56 \ RNN: 2.31, 1.78 \ MLP: 2.40, 1.85],
  [@lstmlit2],[Salt Lake City, Utah, USA \ Austin, Texas, USA],[Public Safety Building, Utah \ Mesowest, East Austin  RAWS Weather Station],[83 days \ One Year],[Temperature, Humidity, Hour, Day of Week, Day of Month, Month Number, Energy Load],[LSTM-MLP],[RMSE (Utah): \ Model 1: 16.9 \ Model 2: 14.1 \ RMSE (Texas): \ Model 2: 45.35],
  [@lstmlit3],[London, England],[London Smart Meters],[Daily, 4 months, 13 months],[Sum Demand, Mean Demand],[RNN, LSTM],[RMSE: \ RNN Short term: 0.02 \ RNN Mid term: 0.03 \ RNN Long term: 0.14 \ LSTM Short term: 0.02 \ LSTM Mid term: 0.03 \ LSTM Long term: 0.15 ],
  [@lstmlit4],[Portugal],[R&D Nester - I-Nergy Project],[Daily],[Hourly load, seasons, holidays, weekend/weekday, time of day],[sNaive, MLP, LSTM, N-Beats, TCN, TFT],[MAPE (%), RMSE: \ sNaive: 6.52, 551.54 \ MLP: 2.44, 211.62 \ LSTM: 2.63, 259.1 \ N-Beats: 2.43, 213.7 \ TCN: 2.52, 226.1, TFT: 3.36, 301.1],
  [@lstmlit5],[Scotland],[Residential Data],[Daily, Weekly],[30 min demand, hourly temp, solar irradiation, precipitation, wind speed, year, month, day, hour, day of week, holidays, time, solar angles and moon phase],[SB-LSTM],[MAE, MAPE(%) \ Day ahead: 0.411, 1.635 \ Week ahead: 0.495, 1.973],
  [@lstmlit6],[Bareilly, India],[Smart Meter Data],[3 minutes],[3-min consumption, time, avg voltage, avg current, grid frequency, hour of day, day of week, season],[LSTM],[MAE, MSE, RMSE \ 0.0013, 0.000008, 0.0028],
  [@lstmlit7],[Finland],[5GNR BS Sites \ University of Oulo],[Hourly],[Power Consumption, Solar PV],[LSTM],[MSE, RMSE \ Consumption: 0.0030, 0.0551 \ Solar PV: 0.0042, 0.0649],
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
  columns: 7,
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
- Add more info, include amount of data, location of dataset
- Include ideally our results should be compared to those in literature that use same dataset
=== Differences in models will generate different results
=== Different features will generate different results
- Add more info, include different features in literature a bit
== Project Novelty
- New methods used on our dataset compared to literature
- Using other data features from dataset compared to literature with same models / methods

#pagebreak()
= Material and Methods
// Harry
== Software

=== Data Storage
The provided datasets were stored in a PostgreSQL database. This has several benefits over CSV. First is column type, by explicitly casting each column to a specific type we get type safety. The second is the ability and strength of SQL joins. As we had several tables that were formatted slightly differently, using SQL made the joins easy. 

=== Data Science
Each team member conducted individual model investigations according to @work-schedule. We each used Jupyter Notebooks, and some of us also used Google Colab. This methodology allowed us to easily share and reproduce results using GitHub. All code was written in Python, and packages used specified in @prog-lang-sect.
#figure(
  table(
    columns: 2,
    table.header([Model], [Assigned To]),
    [Linear Regression], [Saba],
    [Tree Based Ensemble Methods], [Nidhi],
    [LSTM], [Cameron],
    [Transformer], [Harry]
  ),
  caption: [Model Investigation Assignment Schedule]
) <work-schedule>

=== Collaboration
Microsoft Teams was used for group coordination and collaboration. A group chat allowed quick questions to be asked and answered, while team channels provided a space for more formal/important communications. In addition weekly team meetings were scheduled and held, both with just the team and also with Wei Tian. 

=== Version Control
The usage of Git was an essential part of ensuring that work was tracked and attributed correctly. In addition GitHub allows cloud based backup and sync of files, which made file sharing easy, and allowed us to track each other's progress.

=== Programming Language <prog-lang-sect>
Python was the primary programming language chosen for our work due to its familiarity, ease of use, and robust ecosystem. In particular the extensibility through the packages in @py-packages.
#figure(
  table(
    columns: 1,
    table.header([Package]),
    [Pandas], [NumPy], [Matplotlib], [Seaborn], [scikit-learn],
    [Optuna], [XGBoost], [TensorFlow], [PyTorch]
  ),
  caption: [Python Packages Used]
) <py-packages>

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
==== Data Transformation
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
Linear regression is one of the mostly applied statistical methods for forecasting electricity demand [43] [44]. It estimates the relationship between a response and one or more explanatory variables by fitting a linear function to the observed data. The coefficients demonstrate the marginal effect of each explanatory variable on demand, while the model minimises the sum of squared errors to provide efficient estimates under the Gauss-Markov assumptions. Compared with machine learning models such as decision trees or ensemble methods, linear regression is often preferred for its interpretability.

==== Data Preparation <lr-dataprep>
Data preprocessing included creation of polynomial temperature terms e.g., average temp2, min temp2, and max temp2. It also included encoding of calendar-based categorical features; season, quarter, weekend, holidays. Interaction features were also included for conditional effects, such as summer × average temperature, Q1 × average temperature, Q3 × sunlight exposure, and Q4 × average temperature. Precipitation was tested but dropped due to statistical insignificance
==== Model Design

The regression model was specified as an ordinary least squares (OLS) multiple linear regression (MLR) to estimate the average daily electricity demand. Let $D_d$ denote the electricity demand on day d. Mathematically, the model can be expressed as seen in @lr-eqn1.
$ D_d=β_0  +∑_(i=1)^k (β_i  X_id )+ε_d $ <lr-eqn1>
In this model, $β_0$ is the intercept, $β_i$ are the regression coefficients, $X_id$ represents the explanatory variables on day d, and $ε_d$ is the random error term assumed to be independently and identically distributed with mean zero and constant variance. For regression analysis, the explanatory variables are grouped into four components,
1. Deterministic terms ($"Det"_d$) for calendar and seasonal influences on demand. It includes indicators for weekend, holiday, seasons, and quarters.
2. Temperature terms ($"Temp"_d$) to add the functional relationship between demand and weather conditions. It includes average temperature, minimum temperature, maximum temperature, and their squared terms to estimate the nonlinear effects, cooling degree days (CD), heating degree days (HD), and sunlight exposure.
3. Historical demand terms ($"DHist"_d$) to represent the short-term persistence of demand. It includes previous day’s minimum and maximum demand.
4. Interaction terms ($"ITerms"_d$) for conditional effects i.e., the influence of temperature or solar exposure may differ across seasons and quarters. It includes summer × average temperature, Q1 × average temperature, Q3 × sunlight exposure, and Q4 × average temperature
Therefore, the regression model can be expressed as follows in @lr-eqn2.
$ D_d  = "Det"_d  + "Temp"_d  + "DHist"_d  + "ITerms"_d  +ε_d $ <lr-eqn2>
Here, $ε_d$ denotes the stochastic error component. The model is estimated using ordinary least squares (OLS) to minimise the sum of squared residuals and to obtain unbiased efficient coefficient estimates under the Gauss-Markov assumptions.

Weekend and public-holiday indicators are helpful to detect behavioral differences in electricity use. Studies report the demand being usually lower on weekends and holidays compared to weekdays [47]. This pattern has also been recognised by AEMO’s operational forecasts [34, 35]. Season and quarter indicators adjust for repeated annual variation in load. Higher demand in winter and summer relates to heating and cooling needs, while spring and autumn are milder with lower demand. Australian studies have shown that seasonal dummies improve forecast accuracy when used alongside temperature variables [48, 49].
Minimum, maximum, and average temperatures are often included with polynomial terms to incorporate non-linear effects [50]. Studies have discussed the importance of including the non-linearity of temperature respective to demand [51]. Studies in Japan, Thailand, and Australia have used MLR with weather and calendar features to gain low forecasting errors (e.g., MAPE < 3%) [46] [45]. Lagged demand and temperature-based predictors such as HDD & CDD are discussed as strong explanatory factors for short-term load forecasting [52] [53]. The motivation to apply OLS MLR in this study was the inspiration from these empirical findings, as well as the need for a computationally efficient method to benchmark against more complex machine learning models.

=== Tree-Based Ensemble Methods
Tree-based machine learning models are commonly used for regression forecasting problems, as explored in @tree-lit-review-table in @tree-ensemble-section. 

Tree-based machine learning models are based on decision trees. Decision trees recursively partition data based on the value of input features, where each internal node of the tree represents a decision based on a specific feature, leading to a subsequent split eventually leading to the leaf nodes that contain a predicted numerical outcome @nw5. An example of a decision tree for a regression problem looks like @eg-decision-tree. 
#figure(
  image("media/Example Decision Tree.png"),
  caption: [Example of Decision Tree for Regression - Image from  @nw21]
) <eg-decision-tree>
Ensemble learning is commonly used with tree-based machine learning models. Ensemble learning combines a number of different models, that usually results in models with less bias and less variance @nw4. The two most popular ensemble learning methods are boosting and bagging. Boosting is where a number of models are trained sequentially, where each model learns from the previous mistakes of the previous models @nw4. Bagging is where a number of models are trained in parallel, where each model learns from a random subset of the data @nw4. The application of boosting is found in gradient boosting decision trees and bagging is found in random forests. 
==== Gradient Boosting Decision Trees

In gradient boosting decision trees, the idea is to have many weak learners that when combined create a strong learner. All trees are connected in series and each subsequent tree or weak learner tries to minimise the error of the previous tree by fitting into the residuals of the previous step @nw4. The final model aggregates the result of each step and eventually a strong learner is created. An example of this is shown in @eg-gradboost.
#figure(
  image("media/Gradient Boost Decision Tree Example.png"),
  caption: [Example of Gradient Boosting Decision Tree - Image from @nw4]
) <eg-gradboost>
The gradient boosting decision tree models will try to minimise a particular loss function. For the purposes of this project, the models will minimise the mean squared error. The mean squared error loss is calculated by the formula:

$ "Mean Squared Error" = (1 / N) * sum_(i = 1)^N (y_i - hat(y)_i)^2 $

where y is the true value and ŷ is the predicted value and N the total number of samples. All metrics are calculated after any normalisation or scaling has been reversed.

In this project, three gradient boosting decision tree models were explored: XGBoost, LightGBM and CatBoost. These models are popular methods to perform regression forecasting and achieve accurate results according to literature @nw13, @nw14, @nw17, @nw18, @nw19 & @nw20 . They are all more advanced algorithms that have been developed on the standard gradient boosting algorithm that outperform the standard algorithm.

Each of these models will be written in Python and make use of extensive Python libraries. 

There are 2 key steps the design for each method will be separated into:  1 -  Data preparation and 2 - Model design. 

===== Data Preparation <gb-dataprep>
====== Feature Engineering
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

There are a number of tunable hyperparameters included for the XGBoost model. The selected hyperparameters for tuning in this project were max_depth, learning_rate, subsample, colsample_bytree, reg_alpha and reg_lambda, selected based on recommendations in literature @nw2.

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

====== Model 2: CatBoost

CatBoost (or Categorical Boosting) is a gradient boosting method that performs well for categorical features. CatBoost modifies the standard gradient boosting algorithm by incorporating ordered boosting and using target statistics for categorial feature encoding @nw6. Ordered boosting is a method that helps reduce over-fitting, by building each new tree while treating each data point as if it’s not part of the training set when the prediction is being calculated. CatBoost also encodes categorical values based on a distribution of the target variable instead of using the target value itself which helps prevent overfitting @nw6. 

To implement this model, the python libraries CatBoost and CatBoostRegressor were used.

#underline[Hyperparameters:]

There are a number of tunable hyperparameters included for the CatBoost model. The selected hyperparameters for tuning in this project were iterations, depth, learning_rate, subsample, colsample_bytree, and l2_leaf_reg, selected as they are the core training parameters for the model @nw9. The hyperparameter descriptions and values for learning_rate, subsample and colsample_bylevel are the same as those in XGBoost. The different hyperparameters for CatBoost include iterations, depth and l2_leaf_reg. 

#underline[Iterations]

Iterations specify the maximum number of trees @nw9, equivalent to n_estimators in XGBoost. In this project, the values for iterations trialled were: 50, 100, 200 and 300. 

#underline[Depth]

Depth defines the depth of the trees @nw9, equivalent to max_depth in XGBoost. Typical values range from 4 to 10 @nw9. In this project, the values for depth trialled were: 2, 4, 6, 8 and 10. 

#underline[l2_leaf_reg]

L2_leaf_reg specifies the coefficient for L2 regularisation term on leaf values @nw9, equivalent to reg_lambda in XGBoost. This penalty discourages large weights in leaves, helping prevent overfitting @nw9. In this project, the values for l2_leaf_reg were: 1, 3 & 5.

===== Model 3: LightGBM

LightGBM is another gradient boosting method that performs well on categorical features. One of the largest points of difference of LightGBM compared to other gradient boosting methods is that it grows trees leaf-wise and selects the leaf that provides the greatest reduction in loss @nw6. Another difference is that this method relies on an efficient histogram-based method to sort feature values and locate the best split that improves both speed and memory efficiency @nw6. Lastly, it implements gradient-based one-side sampling that focuses on the most informative data samples during training and hence speeds up model training. 

To implement this model, the python libraries LightGBM and LGBMRegressor were used.

#underline[Hyperparameters:
]

There are a number of tunable hyperparameters included for the LightGBM model. The selected hyperparameters for tuning in this project were n_estimators, max_depth, learning_rate, subsample, colsample_bytree, num_leaves and min_data_in_leaf, selected as they are the major training parameters for the model @nw10. The hyperparameter descriptions and values for n_estimators, max_depth, learning_rate, subsample and colsample_bylevel are the same as those in XGBoost. The different hyperparameters for LightGBM include num_leaves and min_data_in_leaf. 

#underline[num_leaves]

num_leaves is the maximum number of leaves in each tree @nw10. Larger values make more complex trees and can improve accuracy but often overfitting is an issue. Lower values make for simpler trees and can underfit. Typical values for this parameter range between 20 and 50 @nw11. In this project, the values for num_leaves trialled were: 20, 30, 40 and 50. 

#underline[min_data_in_leaf]

min_data_in_leaf are the minimum number of data points allowed in a leaf @nw10. This parameter helps avoid overfitting. Typical values for this parameter range between 10 to 100 @nw10. In this project, the values for min_data_in_leaf trialled were: 10, 20, 30, 50 and 100.

==== Gradient Bagging Decision Trees
In gradient bagging decision trees, the idea is to use collective knowledge of several decision trees to make accurate predictions @nw16 by aggregating predictions from numerous decision trees and thereby reducing variance and improving generalisation @nw13. Gradient bagging decision trees make multiple decision trees that are trained on subsets of data. Each tree is built using a random selection of features @nw13. The final prediction is obtained by averaging the outputs @nw13. An example of this is shown in @eg-gradbag.

#figure(
  image("GradBaggExample.png"),
  caption: [Example of Gradient Bagging Decision Tree - Image from @nw15]
) <eg-gradbag>

In this project, the gradient bagging decision tree model explored is Random Forest. Random Forest is a popular method to perform regression forecasting and achieve accurate results according to literature @nw12, @nw13, @nw14 and @nw18. 

===== Data Prepration
The data feature engineering and scaling and normalisation for Random Forest is the same as for gradient boosting decision trees, as described in Section 3.5.2.1.1. 

===== Model Design

Random Forest is a robust and versatile machine learning technique known for its ability to handle regression challenges across multiple domains @nw16. The Random Forest regression approach can handle big datasets of various data types including continuous and categorical variables, can handle non-linear correlations, and due to its ensembling and aggregation approach is robust against overfitting when making predictions. These factors make Random Forest an ideal for load forecasting @nw12. 

To implement this model, the python libraries sklearn and RandomForestRegressor were used.

#underline[Hyperparameters:]

There are a number of tunable hyperparameters included for the Random Forest model. The selected hyperparameters for tuning in this project were n_estimators, max_depth, min_samples_split, min_samples_leaf and max_features, selected based on recommendations in literature @nw22. 
The hyperparameter description and values for n_estimators is the same as those in XGBoost in Section 3.5.2.1.2.1. The different hyperparameters for Random Forest include max_depth, min_samples_split, min_samples_leaf and max_features. 

#underline[max_depth:]

The description of max_depth is the same as those in XGBoost in Section 3.5.2.1.2.1. However, the typical values for max_depth for Random Forest are 5-20 according to literature @nw22. In this project, the values for max_depth trialled were: None (no limit), 5, 10 & 20.

#underline[min_samples_split:]

min_samples_split is the minimum number of samples needed to split a node @nw22. A lower number lets the tree split on small subsets of data, allowing it to capture fine-grained patterns @nw22, but often reduces training generalisation and is at risk of overfitting. A higher number of samples per split results in less splits, creating more conservative trees and reduces overfitting @nw22. Typical values for this are between 2 and 10 @nw22. In this project, the values for min_samples_split trialled were: 2, 5 & 10.

#underline[min_samples_leaf:]

min_samples_leaf is the minimum number of samples required in each leaf node @nw22. For example, having a minimum number of x leaves would mean a decision rule is only considered if it would have at least x samples on each side of the split @nw22. A higher number makes the tree base-learners more general and reduces overfitting, whereas a lower number can improve accuracy and variance but may be at risk of overfitting. Common values for this parameter are between 1 to 10 @nw22. In this project, the values for min_samples_leaf trialled were: 1, 2, 5 & 10. 

#underline[max_features:]

max_features controls how many features or predictor columns are considered when creating a split @nw22. By limiting the number of features per split, this tries to de-correlate trees by ensuring different trees explore different patterns @nw22. A higher value of features makes the trees more similar to one another, which can lead to overfitting whereas lower numbers promote more randomness and diversity which reduces overfitting but can also underfit if too low. Typical values for regression are: None (meaning all features), sqrt (meaning the square root of the total number of features) and log2 (meaning the base-2 value of the total number of features) @nw22. In this project, the values trialled were: None, sqrt and log2. 


=== Long Short-Term Memory Network <lstm_desc>

Long Short-Term memory (or LSTM for short) was derived in 1997 by German researchers Sepp Hochreiter and Jurgen Schmidhuber @lstm2 as a way to mitigate the Vanishing Gradient Problem found in more traditional recurrent neural networks (or RNNs)@lstm1.  They also, unlike RNNs, allow for longer memory processing, which makes them better suited for time series prediction @lstm3.  As can be seen below (@lstm-arc1), the architecture of a LSTM unit includes 3 gates: a forget gate, an input gate and an output gate along with a memory cell.
#figure(
  image("media/lstm2.png", height: 40%),
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

The base architecture for the LSTM can be seen in @lstm-arc2.  The tanh activation function is primarily used coming out of the LSTM layer into the fully connected layer as seen  in other successful studies @lstmlit9 @lstmlit6.  The linear activation into the output layer is also standard.
#figure(
  image("media/lstm3.png", height: 20%),
  caption: [Basic LSTM network architecture]
) <lstm-arc2>
A stacked LSTM structure and a convolutional neural network (CNN) - LSTM hybrid layer will also be explored due to documented success in other research @ref2 @lstmlit5.  The architectures of these models are shown in @lstm-arc3 below.
#figure(
  image("media/lstm4.png", height: 25%),
  caption: [Variant LSTM network architectures]
) <lstm-arc3>

The LSTM functionality will be set up by using the Tensorflow Keras library due to it's simplicity of implementation.  The Adam optimiser was chosen as it is the default optimiser for deep learning models due to it's stability and robustness @ref33.

===== Hyperparameters

There are a number of tunable hyperparameters included for the LSTM model.   In the model definition, this is num_lstm_nodes, and will be extended to include dropout and recurrent_dropout in later experiments.  For the optimiser, the tunable hyperparameter is learning_rate, and when fitting the model the tunable hyperparameter is batch_size.  The number of days in the input matrix (window_size) will be adjusted experimentally, however 7 was chosen as a default as it covers a week of data which should pick up variations in demand.  Hyperparameter tuning will be performed iteratively by the Python Optuna package as described in @model-opt.

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
- Description of modelling method
- Include a diagram? If possible about what the method is
- Refer to literature where and why these methods have been used and why we decided to use them
- Include model implementation specifics
   - Were there certain parameters you had to define? Did you experiment with different values for these parameters? How did you select which parameter values you were going to test? (refer to literature)
- Include code snippets???

The Transformer network architecture (@trans-arc) was introduced in 2017 by researchers at Google @google1. It was designed to replace and outperform the primarily recurrence based models used at the time, both in increased performance and reduced training cost due to parallelisation @transformer2. The architecture is a specific instance of the encoder-decoder models that had become popular in the years prior @transformer1. The primary advancement from this architecture was in the space of natural language processing (NLP), with a myriad of models being developed and becoming familiar in the mainstream such as ChatGPT @transformer1. However, this architecture can also still be applied to forecasting problems @transformer2. 
#figure(
  image("media/transformer.png", height: 40%),
  caption: [Transformer Architecture introduced by @google1]
) <trans-arc>
The novelty of this method lies in the architecture's removal of recurrence entirely, instead relying entirely on attention mechanisms. Attention excels at learning long range dependencies, which is a key challenge in many sequence transduction tasks @google1. A self attention layer connects all positions with a constant ($OO(1)$) number of operations, executed sequentially. In comparison, a recurrent layer requires $OO(n)$ sequential operations. This means that self attention layers are faster than recurrent layers whenever the sequence length $n$ is less than the model dimensionality $d$. 

The transformer takes an input, which in NLP is a sentence or phrase, that is first converted to numbers by an embedding layer before being passed to the encoder portion of the transformer @transformer2. Sequentiality is captured through positional encoding. In our task, we aim to input sequential demand and temperature data, and output a prediction for the next 24 hours of electricity demand.

==== Input
The input can be further broken down between historical data and contextual data. Historical data is the actual temperature and demand recordings. Contextual data is that which can be extracted from the date/time, such as day of the week and month of the year. 

Scaling...\
Positional Encoding...

==== Structure
In essence, the transformer is just another form of neural network. As our task is sequential prediction of only one value at a time, we can simplify the architecture introduced in @google1 and eliminate the need to refeed the already generated outputs back into the model. In addition PyTorch provides an implementation of the attention and feedforward mechanism outlined in @google1 called TransformerEncoderLayer @pytorch1. This allows us to create a straightforward structure as shown in @imp-trans-arc.
#figure(
  image(
    "media/Transformer Architecture.png",
    height: 30%
  ),
  caption: "Implemented Transformer Architecture"
) <imp-trans-arc>
The novelty lies in allowing the attention mechanism (need more understanding) to capture a wide timeframe in training. 

==== Output
The output of the model is a simple floating point estimation of the total demand over the next 24 hours (Note that this is equivalent to the average demand multiplied by 24).

Inverse Scaling...

==== Evaluation
To avoid the effects of variance caused by the inherent randomness present in weight initialisation and dropout, each model was trained 5 times, and the median performing model was chosen as an accurate and reproducible representation of the hyperparameters.

==== Extension
In simplifying the model architecture, the ability (and strength) of the model to perform recursive forecasting (by feeding predicted output back into the model) has been removed. Adding this back in could be a valuable technique in improving accuracy. For example, if the dataset were maintained at 30 minute intervals, and the model were asked to predict the next 24 hours demand at these intervals, then it may be possible to achieve greater accuracy, or results that are of more value to stakeholders.
In addition, the model should be tested on more data streams, the most applicable of which would be other states of Australia. 

== Model Optimisation Techniques <model-opt>
Both the gradient boosting-based and the neural network-based machine learning models require some kind of hyperparameter tuning to be completed to result in the most optimal solution.  

While this can be done manually, this process would be labour intensive and time consuming.  A widely used alternative in much documentation is the GridSearchCV, which looks at every combination of hyperparameters and thus locates the best one @optuna3.  Although it finds the optimal solution with absolute certainty, it also takes a lot of time and computational resources.  Alternatively, RandomSearchCV selects random combinations and finds the best combination out of those attempted @optuna3.  It is less likely to find the most optimal solution, but also takes a lot less time and resources.

Recent literature has shown the rise of a Python library known as Optuna, which assists with automated hyperparameter tuning.  It uses the method of Bayesian Optimisation @optuna3 to move through sets of hyperparameters in a probabilistic way that by definition should improve each iteration, or adjust accordingly.  It is widely used in the last few years due to its platform independence, ease of integration and the ability to visualise the tuning process @optuna2.  A study by Hanifi et al @optuna1 showed Optuna to be the most efficient technique amongst three packages designed for a similar purpose.

== Model Evaluation & Analysis
- How are we evaluating the models? Include descriptions
- Overfitting???

=== Accuracy
Accuracy of each model was determined using mean absolute percentage error (MAPE), defined in @mape.
$ "MAPE" = 1/N sum_(i=1)^N abs(y_i - hat(y)_i)/y_i $ <mape>


#pagebreak()
= Exploratory Data Analysis <eda_section>
// Saba
#emph[This is where you explore your data using histograms, scatterplots, boxplots, numerical summaries, etc.]

From Wei:
- Purpose is to show what's in the data - descriptive plots of key variables
- Want to use results from EDA to guide analysis and decision making
- EG non linear relationship may require transformation
- May want to pay extra attention to interesting patterns
- Select interesting/important ones, rest in the appendix.


@python1 presents the time series of daily average electricity demand (MW) in NSW from 2016 to 2019. It shows prominent seasonal variation i.e., higher demand in summer (December–February) and mid-winter (June–August), and lower demand in the transitional months. This correspond to greater electricity use for air conditioning in hot periods and for heating in colder months; a pattern consistently reported in electricity demand studies for Australia @ref19. System operators such as AEMO account for these seasonal and calendar effects by including month and day-type indicator variables in their forecasting models @ref20.
The series also shows short-term fluctuations caused by weather variation, daily activity, and discrete events such as heatwaves. Extreme conditions in January–February and mid-winter lead to noticeable increase in demand. However, seasonal cycle remains the dominant feature of the series. Across the study period, there is no clear upward/ downward trend, which refers to stable aggregate demand in NSW. In Australian forecasting practice, this stability is usually addressed by treating the trend component as weak, while seasonal and weather-related factors are given greater importance @ref30. Since this study also considers lagged demand features (minimum and maximum 30-minute daily demand), the time series of these variables is presented in Appendix 1.

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
#figure(
image(
  "python code 1.png",
  height: 40%
),
caption: "Daily average 30-min electricity demand (mW)(2016-2019)"
)<python1>


The time series plots of the climate variables show a notable seasonal variability, which is essential for demand forecasting (see Figure 2). The average daily temperature (°C) shows distinct cycles across months. The series shifts from higher values in summer, declining into winter, and then rising again. This seasonal behavior indicates heating and cooling needs, which directly influence energy demand. Ahmed et al. (2012) @ref31 also reported that rising temperature increases electricity demand in NSW, especially in summer and spring.
Precipitation follows an irregular pattern with high peaks on certain days, while many days record low or near-zero rainfall. These abrupt changes suggest that rainfall may act more as an external shock variable in demand models rather than a smooth seasonal factor. Sunlight exposure demonstrates a seasonal cycle i.e., higher values in summer months and reduced levels in winter. This aligns with expected daylight variation and directly affects both solar energy generation and cooling-related demand. Eshragh et al. (2021) @ref26 report that solar exposure and temperature minima/maxima significantly improve forecast accuracy for state power demand including NSW.
Based on these observations, minimum and maximum 30-minute daily temperatures have also been included in this study. In addition, cooling degree (CD) and heating degree (HD) variables are computed and used for forecasting. The time series of these features are presented in Appendix 1.

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
#figure(

  
image("Python Code 2.png", height: 35%),
caption: "Time series plots of climate variables (2016-2019)"
)<python2>

In terms of variation in demand based on type of days, boxplot has been presented for weekday distribution of electricity demand (see Figure 3). It shows a systematic difference between weekdays and weekends. Median demand is consistently higher on weekdays, ranging around ~8000 MW on Monday to Tuesday. By contrast, weekend medians are lower (~7500 MW) for Saturday to Sunday. This reduction points to the effect of reduced commercial and industrial activity during weekends.
Across weekdays, Tuesday records the highest median demand, while Friday shows a slight decrease, although still well above weekend levels. The interquartile range (IQR) is relatively stable across weekdays, which shows similar variability in typical daily loads. Few outliers are also present across all weekdays, likely corresponding to extreme weather events or unusual system conditions.
Weekend demand not only has lower medians but also lower maxima compared to weekdays (e.g., ~9000 MW on Saturday vs ~10,000 MW on Tuesday). This confirms a weekday–weekend effect in NSW demand, consistent with operational expectations where business and industrial loads dominate during weekdays, while residential usage is more prominent during weekends. Earlier NSW and Australian load-forecasting research explicitly recognizes the weekday/weekend effect, for example, Koprinska et al. (2012) @ref32 used separate weekday-based models for NSW load data. 
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
#figure(
image("python code 3.png", height: 35%),
  
caption: "Variation in average electricity demand_t (MW) across weekdays"
)<python3>


In term of monthly variation, electricity demand in NSW shows systematic fluctuations i.e., the seasonal cycle (refer to Figure 4). The highest median demand occurs in winter, with June and July around ~8700 MW recording sustained high levels alongside wide interquartile ranges. It is consistent with heating requirements during colder conditions. Summer months also presented high demand e.g., January and February both with more than 8000 MW. These months showed elevated medians and greater variability. Maximum loads in these months exceed 10,000 MW, which aligns with extreme heat events and widespread air-conditioning use.
In contrast, demand is noticeably lower in the transitional months of autumn and spring. Median decline from March to April (~7500 MW), and reached its lowest in October and November. The narrower interquartile ranges in these months indicates to stable demand when temperatures are milder and heating or cooling is less necessary. Outliers are most frequent in February, May, December, and July, corresponding to months when weather extremes i.e., heatwaves or cold spells, disrupt typical load patterns. This illustration further confirms the seasonal nature of demand in NSW. This confirms the importance of using seasonal effects when modelling electricity demand, a practice well established in Australian forecasting studies @ref31,@ref32.
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
#figure(
image("python code 4.png", height: 35%),  
caption: "Variation in average electricity demand_t (MW) across months"
)<python4>

Lastly, the seasonal cycle from the quarterly boxplots confirm that it is directly relevant for day-ahead forecasting for demand in NSW (see Figure 5). Median demand is highest in the first quarter and third quarter i.e., summer and winter respectively. In contrast, demand is lowest in the fourth quarter, which is consistent with the milder conditions of spring. 
This seasonal structure has two implications for forecasting. First, the broad differences in medians and interquartile ranges across quarters shows that load distributions shift systematically with the season. A forecasting model that does not control for seasonality may misestimate demand when transitioning between summer/winter peaks and spring/autumn troughs. Second, the presence of outliers in specific quarters, such as extreme peaks above 10,000 MW in Q1 (summer) and a cluster of anomalies in Q4 (spring/early summer) indicates that rare weather shocks occur within a season. These can be better anticipated when quarter or season indicators are included alongside weather variables. Therefore, incorporating quarter (or seasonal) dummy variables will help the model differentiate baseline demand levels throughout the year. For example, a hot day in January (Q1) is not treated as equivalent to a hot day in October (Q4), since the baseline seasonal demand is different. 
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
#figure(
image("python code 5.png", height: 35%),
caption: "Electricity demand by Quarter"
)<python5>

To study the predictive power of previous-day minimum and maximum demand for day-ahead forecasting in NSW, the scatter plots between these variables has been presented in Figure 6. Including lagged features is a common approach in short-term load forecasting because electricity demand has temporal dependence, and past demand provides information about current conditions. S
The plot shows a strong positive association in both cases. A more dispersed pattern of points for min demand compared to max demand’s fitted line indicates that min demand may only able to get part of the variability from the prior day to forecast current demand. Because minimum load is influenced by overnight activity and off-peak consumption, which are less stable predictors of daily averages. 
While, the maximum demand from the previous day may provide a more reliable estimate for average demand on the current day, as indicated 4T strong positive fitted line. This stability may be due to the fact that peak demand refers to periods of high system stress, caused largely by climate conditions (e.g., high temperatures leading to cooling load, or low temperatures leading to heating load), which tend to persist across consecutive days.

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
    r2 = model.rsquared
    #for equation text
    ax.text(0.05, 0.95,  f"y = {intercept:.2f} + {slope:.2f}x\nR² = {r2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    ax.set_xlabel(xlabel, fontsize=9) # Labels
    if i == 0:
        ax.set_ylabel(r"Demand$_{t}$ (MW)", fontsize=9)
    else:
        ax.set_ylabel("")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=8)


```
#figure(
image("python code 6.png", height: 32%),

caption: "Scatter plot of demand_t (MW) vs. lagged demand (MW) features"
)<python6>
Minimum, maximum, and average temperatures are included in forecasting because they affect electricity use in different ways. Minimum temperature shows night-time conditions that influence heating needs. Maximum temperature indicates daytime heat stress, which influences cooling demand. As presented in Figure 7, a polynomial fit is used for temperature-based features because the link between demand and temperature is not linear. Demand rises when days are very hot or very cold, while it stays lower at mild temperatures. Studies on Australian load forecasting, such as Hyndman and Fan (2010) @ref22 and Taylor and McSharry (2007) @ref31, also used quadratic terms to model this curved response.
The scatter plots confirm this non-linear relation. Demand falls at mid-range temperatures but increases when the day is too warm or too cold. Among the predictors, average temperature is the strongest (R^2 = 0.54). Maximum and minimum temperatures explain less (R^2 = 0.31 and 0.29). This implies that in day-ahead forecasting for NSW, average temperature is more important, and minimum & maximum temperature are useful for extreme weather periods.

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
    r2 = model.rsquared
    # Generate smooth curve
    x_range = np.linspace(bxp[xvar].min(), bxp[xvar].max(), 200)
    y_fit = intercept + beta1 * x_range + beta2 * x_range**2
    ax.plot(x_range, y_fit, color="red", linewidth=1)
    # Equation text
    ax.text(0.05, 0.95,
            f"y = {intercept:.2f} + {beta1:.2f}x + {beta2:.2f}x²\nR² = {r2:.2f}",
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
#figure(
image("python code 7.png", height: 20%),

caption: "Scatter plot of demand_t (MW) vs. temperature-based features"
)<python7>
Lastly, the scatterplots presented in Figure 8 are presented to observe the relationship between electricity demand and climate-based features. Precipitation shows weak relationships with demand. For daily average demand versus precipitation, estimated line shows almost no effect (R² ≈ 0). This may occur because rainfall does not directly influence electricity use. Sunlight exposure also shows a low effect (R^2 = 0.02). Cooling degree (CD) and heating degree (HD) are stronger predictors, as indicated by R^2 of 0.15 and 0.16, respectively. These features are linked with energy requirements for air conditioning and heating, directly relating weather extremes to electricity consumption. 

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
    r2 = model.rsquared
    ax.text( 0.05, 0.95, f"y = {intercept:.2f} + {slope:.2f}x\nR² = {r2:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")) 
    ax.set_xlabel(xlabel, fontsize=9)
    if i == 0:
        ax.set_ylabel(r"Demand$_{t}$ (MW)", fontsize=9)
    else:
        ax.set_ylabel("")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(axis="both", labelsize=8)




```
#figure(
image("python code 8.png", height: 37%),


caption: "Scatter plot of demand_t (MW) vs. rest of the climate features"
)<python8>


#pagebreak()
= Analysis and Results

== Model Performance
@modelcomp shows the best performing model from each of the methodologies. 
#figure(
  table(
  columns: 4,
  table.header([Model], [Features], [Notes], [MAPE]),
  [Linear regression], [], [], [],
  [XGBoost], [], [], [],
  [LSTM], [Demand, Temperature], [Using Sum Demand, All Temp Features including Temp^2, Seasons & Weekend/Weekday + 75/25 sequential data split], [2.76%],
  [Transformer], [All], [Using Full Dataset + 80/20 sequential data split], []
  ),
  caption: [Model Performance Comparison]
) <modelcomp>

=== Model Specific Results
==== Linear Regression

#figure(
 image("Regression model performance.png", width: 9),
  caption: [Results of Regression]
)
#figure(
image("Forecast on train actual vs predict.png"),
caption: [Forecast plots on train and test demand series using regression model]
)
==== Tree-Based Ensemble Methods 
- Show results from some various different experiments you did & best performing (Table maybe?)
- Figure - predicted vs. actuals graph
- Feature importance
- Overfitting

===== XGBoost

#figure(
  {
    show table.cell: set text(size: 8pt)
    show emph: set align(center)
  table(
    columns: 5,
    table.header([Model Parameter / Variable], [Trial Values], [Best Performing Value], table.cell(colspan:2, grid(
      columns: 2,
      gutter: 7pt,
      grid.cell(colspan:2,[Best Performing Model]), [Train Metrics], [Test Metrics]
    ))),
    [*XGBoost (only day lag demand)*], [], [], [MAPE=0.80%,\ RMSE=4114.18,\ MAE=3027.23,\ R#super()[2]=0.9866], [MAPE=2.08%,\ RMSE=11584.73,\ MAE=7854.73,\ R#super()[2]=0.9042],
    [#underline[Data Feature Selection]], [], [], [], [],
    [_n_estimators_], [100, 250, 500, 750, 1000], [1000], [], [],
    [_max_depth_], [2, 4, 6, 8, 10], [4], [], [],
    [_learning_rate_], [0.02, 0.05, 0.1, 0.15, 0.2], [0.02], [], [],
    [_subsample_], [0.6, 0.7, 0.8, 0.9], [0.6], [], [],
    [_colsample_bytree_], [0.6, 0.7, 0.8, 0.9], [0.8], [], [],
    [_reg_alpha_], [0, 0.1, 0.2, 0.5, 1.0], [1.0], [], [],
    [_reg_lambda_], [0, 0.5, 0.8, 0.9, 1.0], [1.0], [], [],
    
    [*XGBoost (day and week lag demand)*], [], [], [MAPE=0.85%,\ RMSE=4325.16,\ MAE=3242.38,\ R#super()[2]=0.9852], [MAPE=2.11%,\ RMSE=11520.80,\ MAE=7975.22,\ R#super()[2]=0.9053],
    [#underline[Data Feature Selection]], [], [], [], [],
    [_n_estimators_], [100, 250, 500, 750, 1000], [1000], [], [],
    [_max_depth_], [2, 4, 6, 8, 10], [2], [], [],
    [_learning_rate_], [0.02, 0.05, 0.1, 0.15, 0.2], [0.1], [], [],
    [_subsample_], [0.6, 0.7, 0.8, 0.9], [0.7], [], [],
    [_colsample_bytree_], [0.6, 0.7, 0.8, 0.9], [0.7], [], [],
    [_reg_alpha_], [0, 0.1, 0.2, 0.5, 1.0], [1.0], [], [],
    [_reg_lambda_], [0, 0.5, 0.8, 0.9, 1.0], [1.0], [], [],
  )},
  caption: [Results of XGBoost]
)


#figure(
 image("XGBoost - subplot.png"),
  caption: [Results of XGBoost]
)

#figure(
 image("XGBoost - feature importance.png"),
  caption: [Feature Importance - XGBoost]
)


#figure(
 image("XGBoost - feature category.png"),
  caption: [Feature Importance by Category - XGBoost]
)

===== CatBoost

#figure(
  image("Screenshot 2025-09-29 002837.png"),
  caption: [Results of CatBoost]
)

#figure(
 image("CatBoost - subplot.png"),
  caption: [Results of CatBoost]
)

#figure(
 image("CatBoost - feature importance.png"),
  caption: [Feature Importance - CatBoost]
)


#figure(
 image("CatBoost - feature category.png"),
  caption: [Feature Importance by Category - CatBoost]
)

===== LightGBM

#figure(
  image("Screenshot 2025-09-29 003041.png"),
  caption: [Results of LightGBM]
)

#figure(
 image("LightGBM - subplot.png"),
  caption: [Results of LightGBM]
)

#figure(
 image("LightGBM - feature importance.png"),
  caption: [Feature Importance - LightGBM]
)


#figure(
 image("LightGBM - feature category.png"),
  caption: [Feature Importance by Category - LightGBM]
)

===== Random Forest

#figure(
  image("Screenshot 2025-09-30 214000.png"),
  caption: [Results of Random Forest]
)

#figure(
 image("RF - subplot.png"),
  caption: [Results of RF]
)

#figure(
 image("RF - feature importance.png"),
  caption: [Feature Importance - RF]
)


#figure(
 image("RF - feature category.png"),
  caption: [Feature Importance by Category - RF]
)

==== Long Short-Term Memory
///- Show results from some various different experiments you did & best performing (Table maybe?)
///- Figure - predicted vs. actuals graph
///- Feature importance
///- Overfitting

The model runs for LSTM were initially performed to discover the optimal feature set for accurate energy demand predictions.  As can be seen in @lstm_res_4sel below, using the 7 day lagged demand window resulted in a test MAPE under 4%.  The creation of the temp_range feature decreased the overfitting of the model. Experimentation with different window sizes decreased the effectiveness of the model, and within the constraints of the experimentation, 7 days remained the optimal window size.

Precipitation and sunlight resulted in worse metrics, so were left out of the feature set for subsequent runs, and showed that temperature was the main driver of predictive accuracy prior to the introduction of seasonality-based features.  Specific to the LSTM structure, it was evidently able to learn the same patterns from the temperature data than it could from the other weather-related features.

Experimentation with seasonality showed that the combination of seasons and weekday/weekend was optimal.  Interestingly, the models learnt nearly identical information from the day of the week features and the weekday/weekend features, so the latter was chosen due to computational considerations.

Consideration of hybrid models was undertaken, but neither the Stacked-LSTM model or the CNN-LSTM hybrid model increased the efficacy of the results.  In fact, the CNN-LSTM model performed the worst out of any experiment.  Attempting an LSTM-Attention model at the end of experimentation resulted in a good prediction, however there were signs of overfitting and the results on the test set were not optimal.

The final optimal model was reached by iteratively introducing the $"temp"^2$ feature, the dropout and recurrent_dropout hyperparameters and then optimising with the Optuna package.  This resulted in a training MAPE of 2.46% and a test MAPE of 2.76%.



#figure(
  {show table.cell: set text(size: 6pt)
  table(
    columns: 7,
    table.header([Model No.],[Train (Val) Test Split], [Features/Changes], [Train MAPE %],[(Val MAPE %)], [Test MAPE %], [Optimised\*],
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
  ))},caption: [Selected Results of LSTM Modelling]
) <lstm_res_4sel>

#figure(
 image("lstm_final_predictions.png"),
  caption: [Results of LSTM]
)

#figure(
 image("LSTM - feature importance.png"),
  caption: [Feature Importance - LSTM]
)

==== Transformer
- Show results from some various different experiments you did & best performing (Table maybe?)
- Figure - predicted vs. actuals graph
- Feature importance
- Overfitting

#figure(
  table(
    columns: 6,
    table.header([Dataset], [Train Test Split], [Features], [Train MAPE], [Test MAPE], [Optimised\*]),
    [2016-2019], [75:25], [Sum Demand, Avg Temp], [3.09%], [3.93%], [No],
    [2016-2019], [75:25], [All except precipitation and sunlight], [1.44%], [3.31%], [No],
    [2016-2019], [80:20], [Sum Demand, Avg Temp], [2.32%], [3.51%], [No],
    [2016-2019], [80:20], [All except precipitation and sunlight], [1.19%], [2.68%], [No],
    
    [2010-2019], [80:20], [Sum Demand, Avg Temp], [2.73%], [3.57%], [No],
    [2010-2019], [80:20], [All except precipitation and sunlight], [1.70%], [2.83%], [Yes],
    [2010-2019], [90:10], [Sum Demand, Avg Temp], [2.53%], [3.80%], [No],
    [2010-2019], [90:10], [All except precipitation and sunlight], [1.32%], [3.06%], [No],
  ),
)
\* Hyperparameters tuned using Optuna. \

=== Initialisation Weights
When the model is initialised, an element of randomness is present in the initialisation of the model weights. This affects the subsequent performance, not greatly, but significantly enough to make comparison between models using optuna difficult, as often the best performing model in optuna seemed to be an outlier of its own parameters, and I was unable to reproduce it. I ran a model with the same parameters 100 times and here are the results:\
Average MAPE over 100 runs: 2.535510824862226\
#let sdev = 0.0956
Standard Deviation of MAPE over 100 runs: #sdev\
Range of MAPE over 100 runs: 0.4858396878334523, from 2.319153005714153 to 2.8049926935476055

Range of MAPE over 100 runs:
#figure(
  image("media/mape histogram.png"),
  caption: [MAPE Histogram]
)
This result, and particularly the range, indicates that in my hyperparameter optimisation I was likely getting an outlier due to initialisation, which was skewing the Bayesian optimisation utilised by Optuna. In practice, the best performing model is selected to be used, however for training purposes we need to ensure we are achieving the best results. If we assume that all models are distributed evenly, then we can modify our Optuna function to train N models and take an average. In this way we can eliminate outliers and improve our optimisation. With a standard deviation of 0.08, in order to have a confidence level of 68% that we are within 0.01 of the mean, we must do 
$ (sigma / 0.01)^2 = (sdev / 0.01)^2 = #calc.pow((sdev / 0.01),2) $
training runs. However for this project, due to hardware and time constraints, we will limit it to 5 runs, and take the median. This will limit the effect of extreme outliers while still ensuring that training is achieved in a reasonable time.

#pagebreak()
= Discussion
//
Put the results you got in the previous chapter in perspective with respect to the problem studied.
== Linear Regression
- Best selected model (what were the results of the experimentation?)
- What was the final architecture of the best performing model, including
  - data features (subheading)
  - model parameters (subheadings)etc.
- Overall summary of model results
- Overfitting
== Tree-Based Ensemble Methods
- Best selected model (what were the results of the experimentation?)
- What was the final architecture of the best performing model, including
  - data features (subheading)
  - model parameters (subheadings)etc.
- Overall summary of model results
- Overfitting
== Long Short-Term Memory
- Best selected model (what were the results of the experimentation?)
- What was the final architecture of the best performing model, including
  - data features (subheading)
  - model parameters (subheadings)etc.
- Overall summary of model results
- Overfitting
== Transformer
- Best selected model (what were the results of the experimentation?)
- What was the final architecture of the best performing model, including
  - data features (subheading)
  - model parameters (subheadings)etc.
- Overall summary of model results
- Overfitting
== Overall
--The inclusion of the days to be predicted weather data in LR and Tree Based could explain any performance gap between the methodologies.

#pagebreak()
= Conclusion and Further Issues
// Cameron
What are the main conclusions? What are your recommendations for the “client”?
What further analysis could be done in the future?

#pagebreak()
#head[References]
#{
  show heading: none
  bibliography(("references.bib", "bib.yml"), style: "ieee", title: [References])
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

== Appendix 2 - Tables <app-2>
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
    [precipitation], [Precipitation amount], [#highlight[need source]],
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
    [1],[60:20:20],[Sum Temp],[],[],[],[No],
    [2],[60:20:20],[Split Overlap],[],[],[],[No],
    [3],[60:20:20],[Remove Overlap, Add Avg Temp],[],[],[],[No],
    [4],[60:20:20],[Add Min & Max Temp],[],[],[],[No],
    [5],[60:20:20],[Add Temp Range (Max-Min), Remove Min & Max Temp],[],[],[],[No],
    [6],[60:20:20],[All Temperature Features excl. CDD/HDD],[],[],[],[No],
    [7],[60:20:20],[Increase Window Suze to 10],[],[],[],[No],
    [8],[60:20:20],[Decrease Window Size to 5],[],[],[],[No],
    [9],[60:20:20],[Increase Window Size to 14],[],[],[],[No],
    [10],[60:20:20],[Window Size -> 7, Add CDD/HDD],[],[],[],[No],
    [11],[60:20:20],[Remove Avg Temp],[],[],[],[No],
    [12],[60:20:20],[Re-add Avg Temp, Add Precipitation],[],[],[],[No],
    [13],[60:20:20],[Remove Precipitation, Add Sunlight],[],[],[],[No],
    [14],[60:20:20],[Re-Add Precipitation],[],[],[],[No],
    [15],[60:20:20],[Remove Precipitation & Sunlight, Add Seasons],[],[],[],[No],
    [16],[60:20:20],[Remove Seasons, Add Weekday/Weekend],[],[],[],[No],
    [17],[60:20:20],[Remove Weekday/Weekend, Add Day Of Week],[],[],[],[No],
    [18],[60:20:20],[Remove Day Of Week, Add Month],[],[],[],[No],
    [19],[60:20:20],[Remove Month, Add Seasons & Weekday/Weekend],[],[],[],[No],
    [20],[90:10],[Switch to Train/Test],[],[N/A],[],[No],
    [21],[90:10],[Remove Weekday/Weekend],[],[N/A],[],[No],
    [22],[90:10],[Add Weekday/Weekend, Remove Seasons],[],[N/A],[],[No],
    [23],[90:10],[Add Seasons, Post-Optuna],[],[N/A],[],[Yes],
    [24],[90:10],[Alt Optimisation],[],[N/A],[],[Yes],
    [25],[90:10],[Adjust Batch Size to 4],[],[N/A],[],[Yes],
    [26],[90:10],[Stacked LSTM Model],[],[N/A],[],[No],
    [27],[90:10],[CNN-LSTM Model],[],[N/A],[],[Yes],
    [28],[90:10],[Introduce Temp^2 to LSTM Model],[],[N/A],[],[No],
    [29],[90:10],[Add Recurrent Dropout],[],[N/A],[],[No],
    [30],[90:10],[Optimise],[],[N/A],[],[Yes],
    [31],[90:10],[Add Attention to LSTM],[],[N/A],[],[No],
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