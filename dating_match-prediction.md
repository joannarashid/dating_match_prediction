------------------------------------------------------------------------

title: “Predicting Speed Dating Matches with Machine Learning” output:
“github_document”

------------------------------------------------------------------------

# Data

## Data Summary

Speed Dating
Dataset:<https://www.openml.org/search?type=data&sort=runs&id=40536&status=active>

Data Description (Kaggle): This data was gathered from participants in
experimental speed dating events from 2002-2004. During the events, the
attendees would have a four-minute “first date” with every other
participant of the opposite sex. At the end of their four minutes,
participants were asked if they would like to see their date again. They
were also asked to rate their date on six attributes: Attractiveness,
Sincerity, Intelligence, Fun, Ambition, and Shared Interests. The
dataset also includes questionnaire data gathered from participants at
different points in the process. These fields include: demographics,
dating habits, self-perception across key attributes, beliefs on what
others find valuable in a mate, and lifestyle information. There are a
total of 123 features and 8378 instances.

\##Libraries and Raw Data

``` r
#setting seed for entire notebook
knitr::opts_chunk$set(cache = T)

options(scipen=999)

library(corrplot)
library(RColorBrewer)
library(Hmisc)
library(corrplot)
library(tidyverse)
library(class)
library(e1071)
library(caret)
library(factoextra)
library(randomForest)
library(gbm)
library(MASS)
library(kknn)
library(car)

data <- read.csv("https://raw.githubusercontent.com/joannarashid/predicting_dating_match/main/speeddating.csv", header = TRUE)
```

## Data Cleaning

``` r
#df <- data %>% select(-starts_with("d_")) #drops cols starting with d-
#cols_remove <- c(1, 3, 6, 7, 11, 65, 66) #manually remove other unnecessary cols
#df <- df[,-cols_remove]
#df['age_diff'] <- abs(df['age_o'] - df['age']) #add in a diff in age column
#df <- df[,-c(2,3)] #remove duplicate age cols 
#Note: we end up with 8378 observations of 59 variables

#Fix cols samerace and match -- change outcomes to 0 or 1
#df$samerace <- ifelse(df$samerace == "b'0'", 0, 1)
#df$match <- ifelse(df$match == "b'0'", 0, 1)

#Data appears to have many NA values -- see which cols have NAs
#colSums(is.na(df)) #expected_num_interested_in_me has 6578 missing vals, so drop column
#df <- df[,-53] #remaining cols have 14% or less of missing vals 

#Impute missing values with median of the column
#medians <- apply(df, 2, median, na.rm = TRUE) #get median of each col
#for (i in colnames(df)){ #impute missing values with median of that col 
#  df[,i][is.na(df[,i])] <- medians[i]
#}
```

``` r
#I could not get line 40 to run. So importing cleaned csv here
df <- read.csv("https://raw.githubusercontent.com/joannarashid/predicting_dating_match/main/speed_dating_data.csv", header = TRUE)
```

## Exploratory Data Analysis

``` r
#See datatypes present in dataframe
str(df) 
```

    ## 'data.frame':    8378 obs. of  58 variables:
    ##  $ wave                         : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ samerace                     : int  0 0 1 0 0 0 0 0 0 0 ...
    ##  $ importance_same_race         : int  2 2 2 2 2 2 2 2 2 2 ...
    ##  $ importance_same_religion     : int  4 4 4 4 4 4 4 4 4 4 ...
    ##  $ pref_o_attractive            : num  35 60 19 30 30 ...
    ##  $ pref_o_sincere               : num  20 0 18 5 10 ...
    ##  $ pref_o_intelligence          : num  20 0 19 15 20 ...
    ##  $ pref_o_funny                 : num  20 40 18 40 10 ...
    ##  $ pref_o_ambitious             : num  0 0 14 5 10 ...
    ##  $ pref_o_shared_interests      : num  5 0 12 5 20 ...
    ##  $ attractive_o                 : num  6 7 10 7 8 7 3 6 7 6 ...
    ##  $ sinsere_o                    : num  8 8 10 8 7 7 6 7 7 6 ...
    ##  $ intelligence_o               : num  8 10 10 9 9 8 7 5 8 6 ...
    ##  $ funny_o                      : num  8 7 10 8 6 8 5 6 8 6 ...
    ##  $ ambitous_o                   : num  8 7 10 9 9 7 8 8 8 6 ...
    ##  $ shared_interests_o           : num  6 5 10 8 7 7 7 6 9 6 ...
    ##  $ attractive_important         : num  15 15 15 15 15 15 15 15 15 15 ...
    ##  $ sincere_important            : num  20 20 20 20 20 20 20 20 20 20 ...
    ##  $ intellicence_important       : num  20 20 20 20 20 20 20 20 20 20 ...
    ##  $ funny_important              : num  15 15 15 15 15 15 15 15 15 15 ...
    ##  $ ambtition_important          : num  15 15 15 15 15 15 15 15 15 15 ...
    ##  $ shared_interests_important   : num  15 15 15 15 15 15 15 15 15 15 ...
    ##  $ attractive                   : int  6 6 6 6 6 6 6 6 6 6 ...
    ##  $ sincere                      : int  8 8 8 8 8 8 8 8 8 8 ...
    ##  $ intelligence                 : int  8 8 8 8 8 8 8 8 8 8 ...
    ##  $ funny                        : int  8 8 8 8 8 8 8 8 8 8 ...
    ##  $ ambition                     : int  7 7 7 7 7 7 7 7 7 7 ...
    ##  $ attractive_partner           : num  6 7 5 7 5 4 7 4 7 5 ...
    ##  $ sincere_partner              : num  9 8 8 6 6 9 6 9 6 6 ...
    ##  $ intelligence_partner         : num  7 7 9 8 7 7 7 7 8 6 ...
    ##  $ funny_partner                : num  7 8 8 7 7 4 4 6 9 8 ...
    ##  $ ambition_partner             : num  6 5 5 6 6 6 6 5 8 10 ...
    ##  $ shared_interests_partner     : num  5 6 7 8 6 4 7 6 8 8 ...
    ##  $ sports                       : int  9 9 9 9 9 9 9 9 9 9 ...
    ##  $ tvsports                     : int  2 2 2 2 2 2 2 2 2 2 ...
    ##  $ exercise                     : int  8 8 8 8 8 8 8 8 8 8 ...
    ##  $ dining                       : int  9 9 9 9 9 9 9 9 9 9 ...
    ##  $ museums                      : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ art                          : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ hiking                       : int  5 5 5 5 5 5 5 5 5 5 ...
    ##  $ gaming                       : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ clubbing                     : int  5 5 5 5 5 5 5 5 5 5 ...
    ##  $ reading                      : int  6 6 6 6 6 6 6 6 6 6 ...
    ##  $ tv                           : int  9 9 9 9 9 9 9 9 9 9 ...
    ##  $ theater                      : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ movies                       : int  10 10 10 10 10 10 10 10 10 10 ...
    ##  $ concerts                     : int  10 10 10 10 10 10 10 10 10 10 ...
    ##  $ music                        : int  9 9 9 9 9 9 9 9 9 9 ...
    ##  $ shopping                     : int  8 8 8 8 8 8 8 8 8 8 ...
    ##  $ yoga                         : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ interests_correlate          : num  0.14 0.54 0.16 0.61 0.21 0.25 0.34 0.5 0.28 -0.36 ...
    ##  $ expected_happy_with_sd_people: int  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ expected_num_matches         : num  4 4 4 4 4 4 4 4 4 4 ...
    ##  $ like                         : num  7 7 7 7 6 6 6 6 7 6 ...
    ##  $ guess_prob_liked             : num  6 5 5 6 6 5 5 7 7 6 ...
    ##  $ met                          : int  0 1 1 0 0 0 0 0 0 0 ...
    ##  $ match                        : int  0 0 1 1 1 0 0 0 1 0 ...
    ##  $ age_diff                     : int  6 1 1 2 3 4 9 6 7 3 ...

``` r
#Notes: all vars are int or num -- no categorical vars -- samerace and match are binary

#View summary stats for all vars 
summary(df)
```

    ##       wave          samerace      importance_same_race importance_same_religion
    ##  Min.   : 1.00   Min.   :0.0000   Min.   : 0.000       Min.   : 1.000          
    ##  1st Qu.: 7.00   1st Qu.:0.0000   1st Qu.: 1.000       1st Qu.: 1.000          
    ##  Median :11.00   Median :0.0000   Median : 3.000       Median : 3.000          
    ##  Mean   :11.35   Mean   :0.3958   Mean   : 3.777       Mean   : 3.646          
    ##  3rd Qu.:15.00   3rd Qu.:1.0000   3rd Qu.: 6.000       3rd Qu.: 6.000          
    ##  Max.   :21.00   Max.   :1.0000   Max.   :10.000       Max.   :10.000          
    ##  pref_o_attractive pref_o_sincere  pref_o_intelligence  pref_o_funny  
    ##  Min.   :  0.00    Min.   : 0.00   Min.   : 0.00       Min.   : 0.00  
    ##  1st Qu.: 15.00    1st Qu.:15.00   1st Qu.:17.65       1st Qu.:15.00  
    ##  Median : 20.00    Median :18.37   Median :20.00       Median :18.00  
    ##  Mean   : 22.47    Mean   :17.41   Mean   :20.27       Mean   :17.47  
    ##  3rd Qu.: 25.00    3rd Qu.:20.00   3rd Qu.:23.26       3rd Qu.:20.00  
    ##  Max.   :100.00    Max.   :60.00   Max.   :50.00       Max.   :50.00  
    ##  pref_o_ambitious pref_o_shared_interests  attractive_o      sinsere_o     
    ##  Min.   : 0.00    Min.   : 0.00           Min.   : 0.000   Min.   : 0.000  
    ##  1st Qu.: 5.00    1st Qu.:10.00           1st Qu.: 5.000   1st Qu.: 6.000  
    ##  Median :10.00    Median :10.64           Median : 6.000   Median : 7.000  
    ##  Mean   :10.68    Mean   :11.83           Mean   : 6.186   Mean   : 7.169  
    ##  3rd Qu.:15.00    3rd Qu.:15.69           3rd Qu.: 8.000   3rd Qu.: 8.000  
    ##  Max.   :53.00    Max.   :30.00           Max.   :10.500   Max.   :10.000  
    ##  intelligence_o      funny_o         ambitous_o     shared_interests_o
    ##  Min.   : 0.000   Min.   : 0.000   Min.   : 0.000   Min.   : 0.000    
    ##  1st Qu.: 7.000   1st Qu.: 5.000   1st Qu.: 6.000   1st Qu.: 4.000    
    ##  Median : 7.000   Median : 7.000   Median : 7.000   Median : 6.000    
    ##  Mean   : 7.356   Mean   : 6.426   Mean   : 6.798   Mean   : 5.542    
    ##  3rd Qu.: 8.000   3rd Qu.: 8.000   3rd Qu.: 8.000   3rd Qu.: 7.000    
    ##  Max.   :10.000   Max.   :11.000   Max.   :10.000   Max.   :10.000    
    ##  attractive_important sincere_important intellicence_important funny_important
    ##  Min.   :  0.00       Min.   : 0.00     Min.   : 0.00          Min.   : 0.00  
    ##  1st Qu.: 15.00       1st Qu.:15.00     1st Qu.:17.65          1st Qu.:15.00  
    ##  Median : 20.00       Median :18.18     Median :20.00          Median :18.00  
    ##  Mean   : 22.49       Mean   :17.40     Mean   :20.26          Mean   :17.46  
    ##  3rd Qu.: 25.00       3rd Qu.:20.00     3rd Qu.:23.26          3rd Qu.:20.00  
    ##  Max.   :100.00       Max.   :60.00     Max.   :50.00          Max.   :50.00  
    ##  ambtition_important shared_interests_important   attractive    
    ##  Min.   : 0.00       Min.   : 0.00              Min.   : 2.000  
    ##  1st Qu.: 5.00       1st Qu.:10.00              1st Qu.: 6.000  
    ##  Median :10.00       Median :10.64              Median : 7.000  
    ##  Mean   :10.67       Mean   :11.83              Mean   : 7.084  
    ##  3rd Qu.:15.00       3rd Qu.:15.69              3rd Qu.: 8.000  
    ##  Max.   :53.00       Max.   :30.00              Max.   :10.000  
    ##     sincere        intelligence        funny           ambition     
    ##  Min.   : 2.000   Min.   : 2.000   Min.   : 3.000   Min.   : 2.000  
    ##  1st Qu.: 8.000   1st Qu.: 7.000   1st Qu.: 8.000   1st Qu.: 7.000  
    ##  Median : 8.000   Median : 8.000   Median : 8.000   Median : 8.000  
    ##  Mean   : 8.291   Mean   : 7.708   Mean   : 8.399   Mean   : 7.584  
    ##  3rd Qu.: 9.000   3rd Qu.: 9.000   3rd Qu.: 9.000   3rd Qu.: 9.000  
    ##  Max.   :10.000   Max.   :10.000   Max.   :10.000   Max.   :10.000  
    ##  attractive_partner sincere_partner  intelligence_partner funny_partner   
    ##  Min.   : 0.000     Min.   : 0.000   Min.   : 0.000       Min.   : 0.000  
    ##  1st Qu.: 5.000     1st Qu.: 6.000   1st Qu.: 7.000       1st Qu.: 5.000  
    ##  Median : 6.000     Median : 7.000   Median : 7.000       Median : 7.000  
    ##  Mean   : 6.185     Mean   : 7.169   Mean   : 7.356       Mean   : 6.426  
    ##  3rd Qu.: 8.000     3rd Qu.: 8.000   3rd Qu.: 8.000       3rd Qu.: 8.000  
    ##  Max.   :10.000     Max.   :10.000   Max.   :10.000       Max.   :10.000  
    ##  ambition_partner shared_interests_partner     sports          tvsports    
    ##  Min.   : 0.000   Min.   : 0.000           Min.   : 1.000   Min.   : 1.00  
    ##  1st Qu.: 6.000   1st Qu.: 4.000           1st Qu.: 5.000   1st Qu.: 2.00  
    ##  Median : 7.000   Median : 6.000           Median : 7.000   Median : 4.00  
    ##  Mean   : 6.796   Mean   : 5.541           Mean   : 6.431   Mean   : 4.57  
    ##  3rd Qu.: 8.000   3rd Qu.: 7.000           3rd Qu.: 9.000   3rd Qu.: 7.00  
    ##  Max.   :10.000   Max.   :10.000           Max.   :10.000   Max.   :10.00  
    ##     exercise          dining          museums            art        
    ##  Min.   : 1.000   Min.   : 1.000   Min.   : 0.000   Min.   : 0.000  
    ##  1st Qu.: 5.000   1st Qu.: 7.000   1st Qu.: 6.000   1st Qu.: 5.000  
    ##  Median : 6.000   Median : 8.000   Median : 7.000   Median : 7.000  
    ##  Mean   : 6.243   Mean   : 7.786   Mean   : 6.986   Mean   : 6.717  
    ##  3rd Qu.: 8.000   3rd Qu.: 9.000   3rd Qu.: 8.000   3rd Qu.: 8.000  
    ##  Max.   :10.000   Max.   :10.000   Max.   :10.000   Max.   :10.000  
    ##      hiking          gaming          clubbing         reading      
    ##  Min.   : 0.00   Min.   : 0.000   Min.   : 0.000   Min.   : 1.000  
    ##  1st Qu.: 4.00   1st Qu.: 2.000   1st Qu.: 4.000   1st Qu.: 7.000  
    ##  Median : 6.00   Median : 3.000   Median : 6.000   Median : 8.000  
    ##  Mean   : 5.74   Mean   : 3.873   Mean   : 5.748   Mean   : 7.682  
    ##  3rd Qu.: 8.00   3rd Qu.: 6.000   3rd Qu.: 8.000   3rd Qu.: 9.000  
    ##  Max.   :10.00   Max.   :14.000   Max.   :10.000   Max.   :13.000  
    ##        tv            theater           movies         concerts     
    ##  Min.   : 1.000   Min.   : 0.000   Min.   : 0.00   Min.   : 0.000  
    ##  1st Qu.: 3.000   1st Qu.: 5.000   1st Qu.: 7.00   1st Qu.: 5.000  
    ##  Median : 6.000   Median : 7.000   Median : 8.00   Median : 7.000  
    ##  Mean   : 5.311   Mean   : 6.778   Mean   : 7.92   Mean   : 6.827  
    ##  3rd Qu.: 7.000   3rd Qu.: 8.000   3rd Qu.: 9.00   3rd Qu.: 8.000  
    ##  Max.   :10.000   Max.   :10.000   Max.   :10.00   Max.   :10.000  
    ##      music           shopping           yoga        interests_correlate
    ##  Min.   : 1.000   Min.   : 1.000   Min.   : 0.000   Min.   :-0.8300    
    ##  1st Qu.: 7.000   1st Qu.: 4.000   1st Qu.: 2.000   1st Qu.:-0.0100    
    ##  Median : 8.000   Median : 6.000   Median : 4.000   Median : 0.2100    
    ##  Mean   : 7.852   Mean   : 5.635   Mean   : 4.336   Mean   : 0.1963    
    ##  3rd Qu.: 9.000   3rd Qu.: 8.000   3rd Qu.: 6.000   3rd Qu.: 0.4300    
    ##  Max.   :10.000   Max.   :10.000   Max.   :10.000   Max.   : 0.9100    
    ##  expected_happy_with_sd_people expected_num_matches      like      
    ##  Min.   : 1.00                 Min.   : 0.000       Min.   : 0.00  
    ##  1st Qu.: 5.00                 1st Qu.: 2.000       1st Qu.: 5.00  
    ##  Median : 6.00                 Median : 3.000       Median : 6.00  
    ##  Mean   : 5.54                 Mean   : 3.179       Mean   : 6.13  
    ##  3rd Qu.: 7.00                 3rd Qu.: 4.000       3rd Qu.: 7.00  
    ##  Max.   :10.00                 Max.   :18.000       Max.   :10.00  
    ##  guess_prob_liked      met              match           age_diff     
    ##  Min.   : 0.0     Min.   :0.00000   Min.   :0.0000   Min.   : 0.000  
    ##  1st Qu.: 4.0     1st Qu.:0.00000   1st Qu.:0.0000   1st Qu.: 1.000  
    ##  Median : 5.0     Median :0.00000   Median :0.0000   Median : 3.000  
    ##  Mean   : 5.2     Mean   :0.04763   Mean   :0.1647   Mean   : 3.643  
    ##  3rd Qu.: 7.0     3rd Qu.:0.00000   3rd Qu.:0.0000   3rd Qu.: 5.000  
    ##  Max.   :10.0     Max.   :8.00000   Max.   :1.0000   Max.   :32.000

``` r
#Notes: people came to the event in 21 waves
#same race and match are binary - either 0 or 1
#hobbies were ranked (generally) from a scale of 1-10
#importance of same race or religion ranked from 1-10
#attribute importance (e.g. attractive, ambitious, etc.) ranked from 1-100
#most people have never met their speed dating partner before
#age_diff ranges from 0-32 with a mean of 3.64 years 

#How many people have not met their speed dating partner before? 
never_met <- sum(df$met == 0)
met <- nrow(df) - never_met
never_met/nrow(df)
```

    ## [1] 0.9571497

``` r
#Notes: Most people (95.7%) have never met their partner before

#How many people ended up matching? 
matched <- sum(df$match == 1)
no_match <- sum(df$match == 0)
matched/nrow(df)
```

    ## [1] 0.1647171

``` r
#Notes: only 16.5% of people matched -- response var outcome is quite skewed 

#Make histogram of all vars -- many vars so do 10 at a time...
hist.data.frame(df[,1:10])
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
hist.data.frame(df[,11:20])
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-3-2.png)<!-- -->

``` r
hist.data.frame(df[,21:30])
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-3-3.png)<!-- -->

``` r
hist.data.frame(df[,31:40])
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-3-4.png)<!-- -->

``` r
hist.data.frame(df[,41:50])
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-3-5.png)<!-- -->

``` r
hist.data.frame(df[,51:58])
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-3-6.png)<!-- -->

``` r
hist_plot <- df %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram(color = "blue")+
  labs(title = ("Histogram All Variables"))+
  theme(plot.title = element_text(face = "bold"))

hist_plot
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-3-7.png)<!-- -->

``` r
#Notes: most distributions are NOT normally distributed 
#Only pref_o_sincere, pref_o_intelligence, pref_o_ambitious, sincere_important, 
#ambition_important, interests_correlate, expected_happy_with_sd_people, and 
#guess_prob_liked are normally distributed 
#All others have a left/right skew 

#See correlations between vars 
cor <- cor(df)
corrplot(cor, type = "upper", 
         method = "color", 
         tl.cex = 0.5, , 
         order = 'FPC', #ordered by first principal component
         mar=c(0,0,2,0),
         title = "Correlation Matrix Ordered by First Principal Component")
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-3-8.png)<!-- -->

``` r
#Notes: predictor variables are not very strongly correlated to one another
#Correlations are in the range of 0-0.6

#View just correlations between match and other var
cor_match <- cor(df[, colnames(df) != "match"], df$match) 
cor_match
```

    ##                                        [,1]
    ## wave                          -0.0174040103
    ## samerace                       0.0130277005
    ## importance_same_race          -0.0489315816
    ## importance_same_religion      -0.0260184809
    ## pref_o_attractive              0.0155529038
    ## pref_o_sincere                -0.0321967379
    ## pref_o_intelligence            0.0136817882
    ## pref_o_funny                   0.0412517890
    ## pref_o_ambitious              -0.0047030215
    ## pref_o_shared_interests       -0.0479526561
    ## attractive_o                   0.2609273206
    ## sinsere_o                      0.1646732797
    ## intelligence_o                 0.1713040589
    ## funny_o                        0.2702563146
    ## ambitous_o                     0.1336898786
    ## shared_interests_o             0.2518164822
    ## attractive_important           0.0147391886
    ## sincere_important             -0.0320801856
    ## intellicence_important         0.0139874149
    ## funny_important                0.0414679266
    ## ambtition_important           -0.0045453523
    ## shared_interests_important    -0.0479588754
    ## attractive                     0.0358535740
    ## sincere                       -0.0027395934
    ## intelligence                   0.0512743252
    ## funny                          0.0028597288
    ## ambition                       0.0113838212
    ## attractive_partner             0.2607453097
    ## sincere_partner                0.1645662608
    ## intelligence_partner           0.1712793134
    ## funny_partner                  0.2703207387
    ## ambition_partner               0.1338828439
    ## shared_interests_partner       0.2518286656
    ## sports                         0.0216866385
    ## tvsports                      -0.0045792197
    ## exercise                       0.0093536542
    ## dining                         0.0338083158
    ## museums                        0.0148789004
    ## art                            0.0314563471
    ## hiking                         0.0240789880
    ## gaming                         0.0132476960
    ## clubbing                       0.0552034309
    ## reading                        0.0202150381
    ## tv                            -0.0142827250
    ## theater                        0.0001510765
    ## movies                        -0.0228365972
    ## concerts                       0.0269417461
    ## music                          0.0230231436
    ## shopping                      -0.0012349831
    ## yoga                           0.0363240929
    ## interests_correlate            0.0308490515
    ## expected_happy_with_sd_people  0.0276490340
    ## expected_num_matches           0.1180742790
    ## like                           0.3053024200
    ## guess_prob_liked               0.2546694715
    ## met                            0.1006082318
    ## age_diff                      -0.0683933981

``` r
#Notes: Most vars are weakly correlated to match (<0.1)
#attractive_o, sincere_o, intelligence_o, funny_o, ambitious_o, shared_interest_o, attractive_partner, 
#sincere_partner, intelligence partner, funny_partner, ambition_partner, shared_interests_partner, 
#and like have relatively stronger correlations (0.15-0.3) but still weak

#Make boxplots of some more strongly correlated vars (> 0.2) 
boxplot <- df %>%
  pivot_longer(- c(match)) %>%
  ggplot(aes(factor(match), value, fill = match)) +
  geom_boxplot() +
  facet_wrap(vars(name), scales = "free_y")+
  labs(title = ("Boxplot All Variables (Sucessful Match = 1)"))+
  theme(plot.title = element_text(face = "bold"))
boxplot
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-3-9.png)<!-- -->

``` r
#Notes: for all plots, the median of the two groups (match vs no match) are different, 
#suggesting that these predictors may have a statistically significant relation to match
```

## Test and Train sets

``` r
n = dim(df)[1];
n1 = round(.20*n) #20/80% split

flags <- sort(sample(1:n, n1));
train <- df[-flags,];
test <- df[flags,];
```

# Predictive Modeling

## Logistic Regression

``` r
log_model = glm(match ~ ., data = train, family = binomial(link = "logit"))
summary(log_model)
```

    ## 
    ## Call:
    ## glm(formula = match ~ ., family = binomial(link = "logit"), data = train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5947  -0.5252  -0.2962  -0.1228   3.3175  
    ## 
    ## Coefficients:
    ##                                   Estimate   Std. Error z value
    ## (Intercept)                   -12.17809676   2.20576910  -5.521
    ## wave                           -0.00864758   0.00711306  -1.216
    ## samerace                       -0.17300559   0.08246562  -2.098
    ## importance_same_race           -0.02237024   0.01727015  -1.295
    ## importance_same_religion        0.02652065   0.01664484   1.593
    ## pref_o_attractive               0.01981252   0.01487682   1.332
    ## pref_o_sincere                  0.01857041   0.01581433   1.174
    ## pref_o_intelligence             0.04029227   0.01609907   2.503
    ## pref_o_funny                    0.03169287   0.01598672   1.982
    ## pref_o_ambitious                0.01670900   0.01553979   1.075
    ## pref_o_shared_interests         0.02293191   0.01603882   1.430
    ## attractive_o                    0.34325980   0.02904491  11.818
    ## sinsere_o                      -0.05745946   0.03483127  -1.650
    ## intelligence_o                  0.04341287   0.04229288   1.026
    ## funny_o                         0.23032760   0.03322751   6.932
    ## ambitous_o                     -0.11416175   0.03266155  -3.495
    ## shared_interests_o              0.16924255   0.02650065   6.386
    ## attractive_important            0.00599150   0.01436347   0.417
    ## sincere_important              -0.00007294   0.01559542  -0.005
    ## intellicence_important          0.02393430   0.01563354   1.531
    ## funny_important                 0.01301336   0.01550632   0.839
    ## ambtition_important            -0.00399485   0.01514662  -0.264
    ## shared_interests_important     -0.00267036   0.01558941  -0.171
    ## attractive                     -0.08683276   0.04004822  -2.168
    ## sincere                         0.01900605   0.03402027   0.559
    ## intelligence                   -0.06776408   0.03464811  -1.956
    ## funny                          -0.02813026   0.04621226  -0.609
    ## ambition                        0.00482517   0.02881499   0.167
    ## attractive_partner              0.20960022   0.03083845   6.797
    ## sincere_partner                -0.08392127   0.03647633  -2.301
    ## intelligence_partner            0.05322825   0.04334083   1.228
    ## funny_partner                   0.15378144   0.03460038   4.445
    ## ambition_partner               -0.12992425   0.03338184  -3.892
    ## shared_interests_partner        0.04285807   0.02901838   1.477
    ## sports                         -0.02946442   0.02101490  -1.402
    ## tvsports                       -0.01653235   0.01870405  -0.884
    ## exercise                       -0.02218840   0.01906318  -1.164
    ## dining                          0.02888778   0.02840780   1.017
    ## museums                        -0.07963086   0.04290110  -1.856
    ## art                             0.08977629   0.03760687   2.387
    ## hiking                          0.00458626   0.01770105   0.259
    ## gaming                          0.01654097   0.01743717   0.949
    ## clubbing                        0.03111687   0.01755309   1.773
    ## reading                         0.02792479   0.02323258   1.202
    ## tv                              0.05045063   0.02142351   2.355
    ## theater                        -0.03583790   0.02494283  -1.437
    ## movies                         -0.04409961   0.02974378  -1.483
    ## concerts                        0.05217553   0.02844237   1.834
    ## music                          -0.04074551   0.03093389  -1.317
    ## shopping                       -0.05241019   0.02061464  -2.542
    ## yoga                            0.01897503   0.01656479   1.146
    ## interests_correlate             0.11447664   0.14425788   0.794
    ## expected_happy_with_sd_people  -0.01362623   0.02491555  -0.547
    ## expected_num_matches            0.06845272   0.01802361   3.798
    ## like                            0.30484785   0.04260887   7.155
    ## guess_prob_liked                0.17728432   0.02477444   7.156
    ## met                            -0.01330387   0.13441077  -0.099
    ## age_diff                       -0.04640522   0.01435771  -3.232
    ##                                           Pr(>|z|)    
    ## (Intercept)                      0.000000033703614 ***
    ## wave                                      0.224087    
    ## samerace                                  0.035913 *  
    ## importance_same_race                      0.195212    
    ## importance_same_religion                  0.111087    
    ## pref_o_attractive                         0.182935    
    ## pref_o_sincere                            0.240284    
    ## pref_o_intelligence                       0.012323 *  
    ## pref_o_funny                              0.047429 *  
    ## pref_o_ambitious                          0.282268    
    ## pref_o_shared_interests                   0.152782    
    ## attractive_o                  < 0.0000000000000002 ***
    ## sinsere_o                                 0.099014 .  
    ## intelligence_o                            0.304665    
    ## funny_o                          0.000000000004154 ***
    ## ambitous_o                                0.000474 ***
    ## shared_interests_o               0.000000000169887 ***
    ## attractive_important                      0.676580    
    ## sincere_important                         0.996268    
    ## intellicence_important                    0.125780    
    ## funny_important                           0.401341    
    ## ambtition_important                       0.791976    
    ## shared_interests_important                0.863993    
    ## attractive                                0.030143 *  
    ## sincere                                   0.576388    
    ## intelligence                              0.050491 .  
    ## funny                                     0.542711    
    ## ambition                                  0.867013    
    ## attractive_partner               0.000000000010703 ***
    ## sincere_partner                           0.021408 *  
    ## intelligence_partner                      0.219397    
    ## funny_partner                    0.000008809615526 ***
    ## ambition_partner                 0.000099395395404 ***
    ## shared_interests_partner                  0.139695    
    ## sports                                    0.160893    
    ## tvsports                                  0.376755    
    ## exercise                                  0.244448    
    ## dining                                    0.309203    
    ## museums                                   0.063432 .  
    ## art                                       0.016976 *  
    ## hiking                                    0.795562    
    ## gaming                                    0.342822    
    ## clubbing                                  0.076274 .  
    ## reading                                   0.229376    
    ## tv                                        0.018527 *  
    ## theater                                   0.150774    
    ## movies                                    0.138167    
    ## concerts                                  0.066590 .  
    ## music                                     0.187778    
    ## shopping                                  0.011010 *  
    ## yoga                                      0.252000    
    ## interests_correlate                       0.427454    
    ## expected_happy_with_sd_people             0.584450    
    ## expected_num_matches                      0.000146 ***
    ## like                             0.000000000000839 ***
    ## guess_prob_liked                 0.000000000000831 ***
    ## met                                       0.921155    
    ## age_diff                                  0.001229 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 5990.8  on 6701  degrees of freedom
    ## Residual deviance: 4299.4  on 6644  degrees of freedom
    ## AIC: 4415.4
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
#training error
log_train_pred = predict(log_model, newdata = train[,-57], type = "response")
log_train_pred = ifelse(log_train_pred > .5,1,0)
log_train_err <- mean(log_train_pred!= train$match)
print(paste0("Logistic Regression Training Error: ", log_train_err ))
```

    ## [1] "Logistic Regression Training Error: 0.138466129513578"

``` r
#testing error
log_test_pred = predict(log_model, newdata = test[,-57], type = "response")
log_test_pred = ifelse(log_test_pred > .5,1,0)
log_test_err <- mean(log_test_pred!= test$match)
print(paste0("Logistic Regression Testing Error: ", log_test_err))
```

    ## [1] "Logistic Regression Testing Error: 0.142601431980907"

``` r
# check logistic regression model

#check for multicollinearity
vif(log_model)
```

    ##                          wave                      samerace 
    ##                      1.166186                      1.079627 
    ##          importance_same_race      importance_same_religion 
    ##                      1.501398                      1.416363 
    ##             pref_o_attractive                pref_o_sincere 
    ##                     24.137626                      8.118784 
    ##           pref_o_intelligence                  pref_o_funny 
    ##                      7.842388                      6.443973 
    ##              pref_o_ambitious       pref_o_shared_interests 
    ##                      5.807198                      6.703596 
    ##                  attractive_o                     sinsere_o 
    ##                      1.556772                      1.839026 
    ##                intelligence_o                       funny_o 
    ##                      2.150780                      1.879099 
    ##                    ambitous_o            shared_interests_o 
    ##                      1.759037                      1.490991 
    ##          attractive_important             sincere_important 
    ##                     23.932790                      8.341828 
    ##        intellicence_important               funny_important 
    ##                      7.643737                      6.577297 
    ##           ambtition_important    shared_interests_important 
    ##                      5.642638                      6.518646 
    ##                    attractive                       sincere 
    ##                      1.863668                      1.452093 
    ##                  intelligence                         funny 
    ##                      1.840712                      1.519751 
    ##                      ambition            attractive_partner 
    ##                      1.650262                      1.728270 
    ##               sincere_partner          intelligence_partner 
    ##                      1.916183                      2.136637 
    ##                 funny_partner              ambition_partner 
    ##                      1.980768                      1.802373 
    ##      shared_interests_partner                        sports 
    ##                      1.767256                      1.974915 
    ##                      tvsports                      exercise 
    ##                      1.725518                      1.440850 
    ##                        dining                       museums 
    ##                      1.556496                      5.033651 
    ##                           art                        hiking 
    ##                      4.753609                      1.353795 
    ##                        gaming                      clubbing 
    ##                      1.377425                      1.258698 
    ##                       reading                            tv 
    ##                      1.249753                      1.922611 
    ##                       theater                        movies 
    ##                      2.167828                      1.845121 
    ##                      concerts                         music 
    ##                      2.404946                      2.049808 
    ##                      shopping                          yoga 
    ##                      1.910088                      1.326347 
    ##           interests_correlate expected_happy_with_sd_people 
    ##                      1.217646                      1.248294 
    ##          expected_num_matches                          like 
    ##                      1.262708                      2.475819 
    ##              guess_prob_liked                           met 
    ##                      1.453733                      1.053262 
    ##                      age_diff 
    ##                      1.056179

``` r
#several vars are over 5, there is some multicollinearity 

#check model assumptions
plot(log_model)
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->![](dating_match-prediction_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->![](dating_match-prediction_files/figure-gfm/unnamed-chunk-6-3.png)<!-- -->![](dating_match-prediction_files/figure-gfm/unnamed-chunk-6-4.png)<!-- -->

## Stepwise Regression

``` r
step_model = step(log_model, trace = FALSE)
summary(step_model)
```

    ## 
    ## Call:
    ## glm(formula = match ~ samerace + importance_same_race + importance_same_religion + 
    ##     pref_o_intelligence + pref_o_funny + attractive_o + funny_o + 
    ##     ambitous_o + shared_interests_o + attractive_important + 
    ##     intellicence_important + funny_important + attractive + intelligence + 
    ##     attractive_partner + sincere_partner + funny_partner + ambition_partner + 
    ##     shared_interests_partner + sports + museums + art + clubbing + 
    ##     reading + tv + theater + concerts + music + shopping + expected_num_matches + 
    ##     like + guess_prob_liked + age_diff, family = binomial(link = "logit"), 
    ##     data = train)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.5906  -0.5279  -0.3008  -0.1238   3.4032  
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error z value             Pr(>|z|)    
    ## (Intercept)              -10.516405   0.568133 -18.510 < 0.0000000000000002 ***
    ## samerace                  -0.173784   0.081659  -2.128             0.033323 *  
    ## importance_same_race      -0.027862   0.016757  -1.663             0.096361 .  
    ## importance_same_religion   0.030565   0.016178   1.889             0.058856 .  
    ## pref_o_intelligence        0.021744   0.005889   3.693             0.000222 ***
    ## pref_o_funny               0.011183   0.006441   1.736             0.082501 .  
    ## attractive_o               0.342553   0.027928  12.266 < 0.0000000000000002 ***
    ## funny_o                    0.222979   0.031910   6.988   0.0000000000027945 ***
    ## ambitous_o                -0.114066   0.029651  -3.847             0.000120 ***
    ## shared_interests_o         0.166677   0.026224   6.356   0.0000000002072767 ***
    ## attractive_important       0.007445   0.003516   2.117             0.034220 *  
    ## intellicence_important     0.025409   0.006523   3.895   0.0000979969555722 ***
    ## funny_important            0.013083   0.006735   1.943             0.052061 .  
    ## attractive                -0.090028   0.036114  -2.493             0.012671 *  
    ## intelligence              -0.060993   0.032382  -1.884             0.059624 .  
    ## attractive_partner         0.210444   0.030559   6.886   0.0000000000057211 ***
    ## sincere_partner           -0.062165   0.032632  -1.905             0.056779 .  
    ## funny_partner              0.150746   0.034232   4.404   0.0000106423288803 ***
    ## ambition_partner          -0.117335   0.031017  -3.783             0.000155 ***
    ## shared_interests_partner   0.043234   0.028608   1.511             0.130723    
    ## sports                    -0.045059   0.016670  -2.703             0.006873 ** 
    ## museums                   -0.077572   0.041354  -1.876             0.060681 .  
    ## art                        0.091234   0.036062   2.530             0.011408 *  
    ## clubbing                   0.036511   0.016680   2.189             0.028603 *  
    ## reading                    0.033055   0.022658   1.459             0.144602    
    ## tv                         0.036619   0.019070   1.920             0.054825 .  
    ## theater                   -0.041412   0.022899  -1.808             0.070530 .  
    ## concerts                   0.058440   0.027130   2.154             0.031232 *  
    ## music                     -0.050182   0.030254  -1.659             0.097182 .  
    ## shopping                  -0.048185   0.018805  -2.562             0.010398 *  
    ## expected_num_matches       0.067533   0.017452   3.870             0.000109 ***
    ## like                       0.312276   0.041925   7.448   0.0000000000000945 ***
    ## guess_prob_liked           0.177114   0.024456   7.242   0.0000000000004416 ***
    ## age_diff                  -0.048296   0.014061  -3.435             0.000593 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 5990.8  on 6701  degrees of freedom
    ## Residual deviance: 4316.1  on 6668  degrees of freedom
    ## AIC: 4384.1
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
#training error
step_train_pred = predict(step_model, newdata = train[,-57], type = "response");
step_train_pred = ifelse(step_train_pred > .5,1,0)
step_train_err <- mean(step_train_pred!= train$match)
print(paste0("Stepwise Logistic Regression Training Error: ", step_train_err))
```

    ## [1] "Stepwise Logistic Regression Training Error: 0.13682482840943"

``` r
#testing error
step_test_pred = predict(step_model, newdata = test[,-57], type = "response")
step_test_pred = ifelse(step_test_pred > .5,1,0)
step_test_err <- mean(step_test_pred!= test$match)
print(paste0("Stepwise Logistic Regression Testing Error: ", step_test_err))
```

    ## [1] "Stepwise Logistic Regression Testing Error: 0.14200477326969"

``` r
# check stepwise reduced logistic regression model

#check for multicollinearity
vif(step_model)
```

    ##                 samerace     importance_same_race importance_same_religion 
    ##                 1.063007                 1.408180                 1.341930 
    ##      pref_o_intelligence             pref_o_funny             attractive_o 
    ##                 1.055860                 1.049016                 1.443392 
    ##                  funny_o               ambitous_o       shared_interests_o 
    ##                 1.742097                 1.450688                 1.468554 
    ##     attractive_important   intellicence_important          funny_important 
    ##                 1.424050                 1.333398                 1.237291 
    ##               attractive             intelligence       attractive_partner 
    ##                 1.512692                 1.599715                 1.698431 
    ##          sincere_partner            funny_partner         ambition_partner 
    ##                 1.530917                 1.944333                 1.560747 
    ## shared_interests_partner                   sports                  museums 
    ##                 1.723906                 1.246322                 4.687125 
    ##                      art                 clubbing                  reading 
    ##                 4.397421                 1.136348                 1.192018 
    ##                       tv                  theater                 concerts 
    ##                 1.522841                 1.835367                 2.192454 
    ##                    music                 shopping     expected_num_matches 
    ##                 1.957833                 1.587537                 1.186232 
    ##                     like         guess_prob_liked                 age_diff 
    ##                 2.401324                 1.418460                 1.021864

``` r
# stepwise regression improved model. No more multicollinearity 

#check model assumptions
plot(step_model)
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->![](dating_match-prediction_files/figure-gfm/unnamed-chunk-8-2.png)<!-- -->![](dating_match-prediction_files/figure-gfm/unnamed-chunk-8-3.png)<!-- -->![](dating_match-prediction_files/figure-gfm/unnamed-chunk-8-4.png)<!-- -->

``` r
#tails on qq plot show there is perhaps some overfitting
```

## PCA

``` r
#cursory check for multidisciplinary with heatmap
heatmap(as.matrix(df), Colv = NA, Rowv = NA, scale="column")
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

``` r
#no high multicolliniearity found

pca = prcomp(train[,-57], center = TRUE)#, scale = TRUE)
summary(pca)
```

    ## Importance of components:
    ##                           PC1     PC2     PC3     PC4    PC5     PC6     PC7
    ## Standard deviation     14.288 13.4058 7.38418 7.34770 7.1341 6.75265 6.59040
    ## Proportion of Variance  0.215  0.1893 0.05742 0.05686 0.0536 0.04802 0.04574
    ## Cumulative Proportion   0.215  0.4042 0.46166 0.51851 0.5721 0.62013 0.66587
    ##                           PC8     PC9    PC10    PC11    PC12    PC13    PC14
    ## Standard deviation     6.3977 6.18444 5.85295 5.26201 4.26030 3.79520 3.75031
    ## Proportion of Variance 0.0431 0.04028 0.03608 0.02916 0.01911 0.01517 0.01481
    ## Cumulative Proportion  0.7090 0.74925 0.78533 0.81449 0.83360 0.84877 0.86358
    ##                           PC15    PC16    PC17   PC18    PC19    PC20    PC21
    ## Standard deviation     3.37681 3.21121 3.02892 2.9397 2.66666 2.53462 2.44079
    ## Proportion of Variance 0.01201 0.01086 0.00966 0.0091 0.00749 0.00677 0.00627
    ## Cumulative Proportion  0.87559 0.88645 0.89611 0.9052 0.91270 0.91947 0.92574
    ##                           PC22    PC23    PC24    PC25    PC26    PC27    PC28
    ## Standard deviation     2.24466 2.17771 2.08572 2.03601 1.92973 1.88072 1.84553
    ## Proportion of Variance 0.00531 0.00499 0.00458 0.00437 0.00392 0.00372 0.00359
    ## Cumulative Proportion  0.93105 0.93604 0.94062 0.94499 0.94891 0.95263 0.95622
    ##                           PC29    PC30    PC31    PC32    PC33    PC34    PC35
    ## Standard deviation     1.73493 1.67489 1.60940 1.59567 1.56087 1.50755 1.48179
    ## Proportion of Variance 0.00317 0.00295 0.00273 0.00268 0.00257 0.00239 0.00231
    ## Cumulative Proportion  0.95939 0.96234 0.96507 0.96775 0.97032 0.97271 0.97502
    ##                           PC36   PC37    PC38    PC39    PC40    PC41   PC42
    ## Standard deviation     1.42526 1.3794 1.35261 1.28353 1.21386 1.20009 1.1917
    ## Proportion of Variance 0.00214 0.0020 0.00193 0.00173 0.00155 0.00152 0.0015
    ## Cumulative Proportion  0.97716 0.9792 0.98109 0.98283 0.98438 0.98590 0.9874
    ##                           PC43    PC44    PC45    PC46    PC47    PC48    PC49
    ## Standard deviation     1.18748 1.13156 1.10141 1.05917 1.05350 1.02811 0.96649
    ## Proportion of Variance 0.00148 0.00135 0.00128 0.00118 0.00117 0.00111 0.00098
    ## Cumulative Proportion  0.98888 0.99023 0.99150 0.99269 0.99385 0.99497 0.99595
    ##                           PC50    PC51    PC52    PC53    PC54    PC55    PC56
    ## Standard deviation     0.92972 0.86448 0.83953 0.79094 0.73690 0.47087 0.27198
    ## Proportion of Variance 0.00091 0.00079 0.00074 0.00066 0.00057 0.00023 0.00008
    ## Cumulative Proportion  0.99686 0.99765 0.99839 0.99905 0.99962 0.99986 0.99993
    ##                           PC57
    ## Standard deviation     0.25205
    ## Proportion of Variance 0.00007
    ## Cumulative Proportion  1.00000

``` r
#scree plot
fviz_eig(pca, ncp = 15)
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-9-2.png)<!-- -->

``` r
#another option for scree plot with red line = 1
screeplot(pca, main = "Scree Plot of Principal Components", type = "line")
abline(h=1, col="red")
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-9-3.png)<!-- -->

``` r
#scree plot not ideal, so additional proportion of variance plot is needed

#get eigenvalues by squaring the standard deviation 
var <- pca$sdev^2

#get proportion of variance
pvar <- var/sum(var)
plot(pvar, xlab = "Principal Component", ylab = "Proportion of Variance",
     ylim = c(0,1) , type= "b")
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-9-4.png)<!-- -->

``` r
#selecting components 1-3 per scree plot
#select_comps <- pca$x[,0:15]

#new data frame with principal components and response data
pca.frame = as.data.frame(pca$x[,1:15])
pca.frame$match = train$match

#pca_df <- data.frame(cbind(select_comps, train$match))

#liner model with selected components
pca_log <- glm(match~., data = pca.frame, family = binomial(link = "logit"))

summary(pca_log)
```

    ## 
    ## Call:
    ## glm(formula = match ~ ., family = binomial(link = "logit"), data = pca.frame)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.1546  -0.5803  -0.3757  -0.1915   2.9179  
    ## 
    ## Coefficients:
    ##               Estimate Std. Error z value             Pr(>|z|)    
    ## (Intercept) -2.1280808  0.0470116 -45.267 < 0.0000000000000002 ***
    ## PC1          0.0014115  0.0024869   0.568             0.570334    
    ## PC2          0.0049592  0.0026822   1.849             0.064467 .  
    ## PC3          0.0196397  0.0049643   3.956    0.000076144496426 ***
    ## PC4          0.0037252  0.0050610   0.736             0.461691    
    ## PC5          0.0204166  0.0051743   3.946    0.000079541949928 ***
    ## PC6         -0.0033056  0.0053850  -0.614             0.539313    
    ## PC7          0.0304931  0.0054740   5.571    0.000000025399487 ***
    ## PC8         -0.0024510  0.0057428  -0.427             0.669533    
    ## PC9         -0.0249910  0.0061154  -4.087    0.000043783401938 ***
    ## PC10         0.0006406  0.0062925   0.102             0.918908    
    ## PC11         0.0240945  0.0069691   3.457             0.000546 ***
    ## PC12         0.1945126  0.0097556  19.939 < 0.0000000000000002 ***
    ## PC13         0.2601771  0.0113703  22.882 < 0.0000000000000002 ***
    ## PC14         0.0129702  0.0100762   1.287             0.198020    
    ## PC15         0.0803505  0.0112202   7.161    0.000000000000799 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 5990.8  on 6701  degrees of freedom
    ## Residual deviance: 4815.1  on 6686  degrees of freedom
    ## AIC: 4847.1
    ## 
    ## Number of Fisher Scoring iterations: 5

``` r
#training error
pca_train_pred <- predict(pca_log, newdata = pca.frame[,-16], type= "response")
pca_train_pred = ifelse(pca_train_pred > .5,1,0)
pca_train_err <- mean(pca_train_pred!= train$match)
print(paste0("PCA Regression Training Error: ", pca_train_err))
```

    ## [1] "PCA Regression Training Error: 0.160399880632647"

``` r
#testing error
pcatest = prcomp(test[,names(test)!= "match"])
pca.testframe = as.data.frame(pcatest$x[,1:15])
pca.testframe$match = test$match

pca_test_pred <- predict(pca_log, newdata=pca.testframe[,-16], type = "response");
pca_test_pred = ifelse(pca_test_pred > .5,1,0)
pca_test_err <- mean(pca_test_pred!= pca.testframe$match)
print(paste0("PCA Regression Testing Error: ", pca_test_err))
```

    ## [1] "PCA Regression Testing Error: 0.219570405727924"

## Additional PCA with 3 PCs

``` r
#new data frame with principal components and response data
pca.frame2 = as.data.frame(pca$x[,1:3])
pca.frame2$match = train$match

#pca_df <- data.frame(cbind(select_comps, train$match))

#liner model with selected components
pca_log2 <- glm(match~., data = pca.frame2, family = binomial(link = "logit"))

summary(pca_log2)
```

    ## 
    ## Call:
    ## glm(formula = match ~ ., family = binomial(link = "logit"), data = pca.frame2)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -0.8454  -0.6124  -0.5809  -0.5562   2.0907  
    ## 
    ## Coefficients:
    ##              Estimate Std. Error z value             Pr(>|z|)    
    ## (Intercept) -1.632471   0.033146 -49.252 < 0.0000000000000002 ***
    ## PC1          0.001598   0.002237   0.714             0.475134    
    ## PC2          0.006292   0.002384   2.639             0.008324 ** 
    ## PC3          0.015199   0.004314   3.523             0.000426 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 5990.8  on 6701  degrees of freedom
    ## Residual deviance: 5971.2  on 6698  degrees of freedom
    ## AIC: 5979.2
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
#training error
pca_train_pred2 <- predict(pca_log2, newdata = pca.frame2[,-4], type= "response")
pca_train_pred2 = ifelse(pca_train_pred2 > .5,1,0)
pca_train_err2 <- mean(pca_train_pred2!= train$match)
print(paste0("PCA Regression Training Error: ", pca_train_err2))
```

    ## [1] "PCA Regression Training Error: 0.164428528797374"

``` r
#testing error
pcatest2 = prcomp(test[,names(test)!= "match"])
pca.testframe2 = as.data.frame(pcatest2$x[,1:3])
pca.testframe2$match = test$match

pca_test_pred2 <- predict(pca_log2, newdata=pca.testframe2[,-4], type = "response");
pca_test_pred2 = ifelse(pca_test_pred2 > .5,1,0)
pca_test_err2 <- mean(pca_test_pred2!= pca.testframe2$match)
print(paste0("PCA Regression Testing Error: ", pca_test_err2))
```

    ## [1] "PCA Regression Testing Error: 0.165871121718377"

## Naive Bayes

``` r
nb_model <- naiveBayes(as.factor(match) ~. , data = train)

#training error
nb_train_err <- mean(predict(nb_model, newdata = train) != train$match)
print(paste0("Naive Bayes Training Error: ", nb_train_err))
```

    ## [1] "Naive Bayes Training Error: 0.214115189495673"

``` r
#testing error
nb_test_pred <- predict(nb_model, newdata = test) != test$match
nb_test_err <- mean(nb_test_pred != test$match)
print(paste0("Naive Bayes Testing Error: ", nb_test_err))
```

    ## [1] "Naive Bayes Testing Error: 0.269689737470167"

## Linear Discriminant Analysis

``` r
lda_model <- lda(train[,-57], train$match)
lda_model
```

    ## Call:
    ## lda(train[, -57], train$match)
    ## 
    ## Prior probabilities of groups:
    ##         0         1 
    ## 0.8355715 0.1644285 
    ## 
    ## Group means:
    ##       wave  samerace importance_same_race importance_same_religion
    ## 0 11.37643 0.3933929             3.836786                 3.694464
    ## 1 10.91652 0.4056261             3.443739                 3.487296
    ##   pref_o_attractive pref_o_sincere pref_o_intelligence pref_o_funny
    ## 0          22.39987       17.45662            20.21172     17.40262
    ## 1          22.74984       16.89373            20.61032     18.10856
    ##   pref_o_ambitious pref_o_shared_interests attractive_o sinsere_o
    ## 0         10.67367                11.94584     5.954357  7.054107
    ## 1         10.63607                11.13381     7.339383  7.777677
    ##   intelligence_o  funny_o ambitous_o shared_interests_o attractive_important
    ## 0       7.239464 6.195000   6.694286           5.310179             22.33004
    ## 1       7.920145 7.582577   7.288113           6.653811             23.10223
    ##   sincere_important intellicence_important funny_important ambtition_important
    ## 0          17.52898               20.22050        17.40223             10.6704
    ## 1          16.94618               20.56353        17.97613             10.4801
    ##   shared_interests_important attractive  sincere intelligence    funny ambition
    ## 0                   11.99520   7.065179 8.294286     7.676250 8.409464 7.573929
    ## 1                   11.06588   7.210526 8.286751     7.877495 8.401089 7.587114
    ##   attractive_partner sincere_partner intelligence_partner funny_partner
    ## 0           5.961679        7.046696             7.243393      6.185714
    ## 1           7.284483        7.806715             7.962341      7.578947
    ##   ambition_partner shared_interests_partner   sports tvsports exercise   dining
    ## 0         6.692054                 5.322232 6.401071 4.559464 6.220357 7.749643
    ## 1         7.300817                 6.659256 6.565336 4.523593 6.265880 7.914701
    ##   museums      art   hiking   gaming clubbing  reading       tv  theater
    ## 0 6.96000 6.680536 5.676429 3.843571 5.686964 7.658214 5.341964 6.773214
    ## 1 7.03176 6.861162 5.873866 3.952813 6.055354 7.774955 5.201452 6.751361
    ##     movies concerts    music shopping     yoga interests_correlate
    ## 0 7.933036 6.799464 7.833036 5.661429 4.283929           0.1947643
    ## 1 7.825771 6.964610 7.921053 5.580762 4.564428           0.2119238
    ##   expected_happy_with_sd_people expected_num_matches     like guess_prob_liked
    ## 0                      5.513036             3.063179 5.879054         4.967321
    ## 1                      5.658802             3.769873 7.378403         6.394283
    ##          met age_diff
    ## 0 0.03357143 3.695000
    ## 1 0.09891107 3.205082
    ## 
    ## Coefficients of linear discriminants:
    ##                                         LD1
    ## wave                          -0.0073046729
    ## samerace                      -0.1048381231
    ## importance_same_race          -0.0168828225
    ## importance_same_religion       0.0146464170
    ## pref_o_attractive              0.0164615735
    ## pref_o_sincere                 0.0148423148
    ## pref_o_intelligence            0.0291033613
    ## pref_o_funny                   0.0276579514
    ## pref_o_ambitious               0.0111400606
    ## pref_o_shared_interests        0.0158268755
    ## attractive_o                   0.2170654709
    ## sinsere_o                     -0.0359974302
    ## intelligence_o                 0.0222998251
    ## funny_o                        0.1355576503
    ## ambitous_o                    -0.0626300818
    ## shared_interests_o             0.1128590300
    ## attractive_important           0.0064823217
    ## sincere_important              0.0039681381
    ## intellicence_important         0.0177700035
    ## funny_important                0.0134888248
    ## ambtition_important            0.0008355461
    ## shared_interests_important    -0.0032164451
    ## attractive                    -0.0352189568
    ## sincere                       -0.0020265306
    ## intelligence                  -0.0528697280
    ## funny                         -0.0052030709
    ## ambition                       0.0107522705
    ## attractive_partner             0.1460088459
    ## sincere_partner               -0.0522246452
    ## intelligence_partner           0.0326761155
    ## funny_partner                  0.0852437689
    ## ambition_partner              -0.0592751958
    ## shared_interests_partner       0.0301993388
    ## sports                        -0.0212761894
    ## tvsports                      -0.0009776773
    ## exercise                      -0.0261638584
    ## dining                         0.0300397236
    ## museums                       -0.0548166399
    ## art                            0.0662858253
    ## hiking                         0.0031176963
    ## gaming                         0.0061245881
    ## clubbing                       0.0228106355
    ## reading                        0.0044453052
    ## tv                             0.0329607854
    ## theater                       -0.0207739657
    ## movies                        -0.0443137317
    ## concerts                       0.0400593067
    ## music                         -0.0236180618
    ## shopping                      -0.0426770391
    ## yoga                           0.0096935808
    ## interests_correlate            0.0613837593
    ## expected_happy_with_sd_people -0.0091747746
    ## expected_num_matches           0.0542989431
    ## like                           0.1549948081
    ## guess_prob_liked               0.1173204254
    ## met                            0.2451368529
    ## age_diff                      -0.0337878098

``` r
#training error
lda_train_err <- mean(predict(lda_model, train[,-57])$class != train$match)
print(paste0("LDA Training Error: ", lda_train_err))
```

    ## [1] "LDA Training Error: 0.142494777678305"

``` r
#testing error
lda_test_pred <- predict(lda_model, test[,-57])$class
lda_test_err <- mean(lda_test_pred != test$match)
print(paste0("LDA Training Error: ", lda_test_err))
```

    ## [1] "LDA Training Error: 0.143198090692124"

## Quadratic Discriminant Analysis

``` r
qda_model <- qda(train[,-57], train[,57]);
qda_model
```

    ## Call:
    ## qda(train[, -57], train[, 57])
    ## 
    ## Prior probabilities of groups:
    ##         0         1 
    ## 0.8355715 0.1644285 
    ## 
    ## Group means:
    ##       wave  samerace importance_same_race importance_same_religion
    ## 0 11.37643 0.3933929             3.836786                 3.694464
    ## 1 10.91652 0.4056261             3.443739                 3.487296
    ##   pref_o_attractive pref_o_sincere pref_o_intelligence pref_o_funny
    ## 0          22.39987       17.45662            20.21172     17.40262
    ## 1          22.74984       16.89373            20.61032     18.10856
    ##   pref_o_ambitious pref_o_shared_interests attractive_o sinsere_o
    ## 0         10.67367                11.94584     5.954357  7.054107
    ## 1         10.63607                11.13381     7.339383  7.777677
    ##   intelligence_o  funny_o ambitous_o shared_interests_o attractive_important
    ## 0       7.239464 6.195000   6.694286           5.310179             22.33004
    ## 1       7.920145 7.582577   7.288113           6.653811             23.10223
    ##   sincere_important intellicence_important funny_important ambtition_important
    ## 0          17.52898               20.22050        17.40223             10.6704
    ## 1          16.94618               20.56353        17.97613             10.4801
    ##   shared_interests_important attractive  sincere intelligence    funny ambition
    ## 0                   11.99520   7.065179 8.294286     7.676250 8.409464 7.573929
    ## 1                   11.06588   7.210526 8.286751     7.877495 8.401089 7.587114
    ##   attractive_partner sincere_partner intelligence_partner funny_partner
    ## 0           5.961679        7.046696             7.243393      6.185714
    ## 1           7.284483        7.806715             7.962341      7.578947
    ##   ambition_partner shared_interests_partner   sports tvsports exercise   dining
    ## 0         6.692054                 5.322232 6.401071 4.559464 6.220357 7.749643
    ## 1         7.300817                 6.659256 6.565336 4.523593 6.265880 7.914701
    ##   museums      art   hiking   gaming clubbing  reading       tv  theater
    ## 0 6.96000 6.680536 5.676429 3.843571 5.686964 7.658214 5.341964 6.773214
    ## 1 7.03176 6.861162 5.873866 3.952813 6.055354 7.774955 5.201452 6.751361
    ##     movies concerts    music shopping     yoga interests_correlate
    ## 0 7.933036 6.799464 7.833036 5.661429 4.283929           0.1947643
    ## 1 7.825771 6.964610 7.921053 5.580762 4.564428           0.2119238
    ##   expected_happy_with_sd_people expected_num_matches     like guess_prob_liked
    ## 0                      5.513036             3.063179 5.879054         4.967321
    ## 1                      5.658802             3.769873 7.378403         6.394283
    ##          met age_diff
    ## 0 0.03357143 3.695000
    ## 1 0.09891107 3.205082

``` r
#training error
qda_train_err <- mean(predict(qda_model, train[,-57])$class != train$match)
print(paste0("LDA Training Error: ", qda_train_err))
```

    ## [1] "LDA Training Error: 0.144583706356312"

``` r
#testing error
qda_test_pred <- predict(qda_model, test[,-57])$class != test$match
qda_test_err <- mean(predict(qda_model, test[,-57])$class != test$match)
print(paste0("LDA Test Error: ", qda_test_err))
```

    ## [1] "LDA Test Error: 0.183174224343675"

## KNN

``` r
#find ideal value for k
loocv_model = train.kknn(match ~ .,
                   train,
                   ks=(1:35),
                   kernel = "optimal",
                   scale=TRUE) #scales the data

plot(loocv_model, main ="Values of K for KNN Model")
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
# k = 26 has the lowest MSE
```

``` r
knn_model <- kknn(match~ .,
             train = train,
             test = test,
             k =  26,
             kernel = "optimal",
             scale = TRUE)

## Testing Error
knn_pred <- predict(knn_model)
knn_pred = ifelse(knn_pred > .5,1,0)
knn_test_err <- mean(knn_pred != test$match)
print(paste0("KNN Test Error: ", knn_test_err))
```

    ## [1] "KNN Test Error: 0.152744630071599"

``` r
knn_train_err = "NA"
```

## Random Forest

``` r
rf_model <- randomForest(as.factor(match) ~.,
                   data = train, 
                   ntree = 500,
                   mtry = 8, #p = 58, default mtry is sqrt(p) = approx. 8
                   nodesize = 2, # response values are either 0 or 1
                   importance = TRUE)

rf.pred = predict(rf_model, test, type="class")

confusionMatrix(data=factor(rf.pred), reference = factor(test$match))
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 1378  208
    ##          1   20   70
    ##                                                
    ##                Accuracy : 0.864                
    ##                  95% CI : (0.8466, 0.88)       
    ##     No Information Rate : 0.8341               
    ##     P-Value [Acc > NIR] : 0.0004386            
    ##                                                
    ##                   Kappa : 0.3257               
    ##                                                
    ##  Mcnemar's Test P-Value : < 0.00000000000000022
    ##                                                
    ##             Sensitivity : 0.9857               
    ##             Specificity : 0.2518               
    ##          Pos Pred Value : 0.8689               
    ##          Neg Pred Value : 0.7778               
    ##              Prevalence : 0.8341               
    ##          Detection Rate : 0.8222               
    ##    Detection Prevalence : 0.9463               
    ##       Balanced Accuracy : 0.6187               
    ##                                                
    ##        'Positive' Class : 0                    
    ## 

``` r
varImpPlot(rf_model)
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
rf_test_err <- mean(rf.pred != test$match)
rf_train_error <- "NA"

print(paste0("Random Forest Test Error: ", rf_test_err))
```

    ## [1] "Random Forest Test Error: 0.136038186157518"

## Gradient Boosting

``` r
gb_model <- gbm(match~ .,
                data = train,
                distribution = 'bernoulli',
                n.trees = 3000, 
                shrinkage = 0.01, 
                interaction.depth = 3, 
                cv.folds = 10)

perf_gbm1 = gbm.perf(gb_model, method="cv") 
```

![](dating_match-prediction_files/figure-gfm/unnamed-chunk-17-1.png)<!-- -->

``` r
perf_gbm1 # = 2904
```

    ## [1] 2995

``` r
#training error
gb_pred <- predict(gb_model,newdata = train, n.trees=perf_gbm1, type="response")
gb_pred[1:10]
```

    ##  [1] 0.27160618 0.49688338 0.78279055 0.78589768 0.37702757 0.25128949
    ##  [7] 0.12475357 0.90373634 0.06142249 0.22406866

``` r
yhat <- ifelse(gb_pred < 0.5, 0, 1)
yhat[1:10]
```

    ##  [1] 0 0 1 1 0 0 0 1 0 0

``` r
gb_train_err <- sum(yhat != train$match)/length(train$match)
print(paste0("Gradient Boosting Training Error: ", gb_train_err))
```

    ## [1] "Gradient Boosting Training Error: 0.100566994926887"

``` r
#testing error
gb_test_pred <- ifelse(predict(gb_model,newdata = test[,-57], n.trees=perf_gbm1, type="response") < 0.5, 0, 1)
gb_test_err <- mean(gb_test_pred != test$match) 
print(paste0("Gradient Boosting Test Error: ", gb_test_err))
```

    ## [1] "Gradient Boosting Test Error: 0.132458233890215"

# Cross Validation

``` r
B=20;
CVALL = NULL

for (i in 1:B) {
  
  #randomly the data into training and test
  finalerror = NULL;
  flags <- sort(sample(1:n, n1));
  train <- df[-flags,];
  test <- df[flags,];
  test[,-57]
  
  #Full Log Regression 
  log_model = glm(match ~ ., data = train, family= binomial(link = "logit"))
  log_test_pred = predict(log_model, newdata = test[,-57], type = "response")
  log_test_pred = ifelse(log_test_pred > .5,1,0)
  log_err <- mean(log_test_pred!= test$match)

  #log regression with step
  step_model = step(log_model, trace = FALSE)
  step_test_pred = predict(step_model, newdata = test[,-57], type = "response")
  step_test_pred = ifelse(step_test_pred > .5,1,0)
  step_err <- mean(step_test_pred!= test$match)
  
  #PCA
  pca = prcomp(train[,-57], center = TRUE)#, scale = TRUE)
  select_comps <- pca$x[,1:15]
  pca_df <- data.frame(select_comps)
  pca_df$match = train$match
  
  pca_log <- glm(match~., data = pca_df, family = binomial(link = "logit"))
  
  pcatest = prcomp(test[,names(test)!= "match"])
  pca.testframe = as.data.frame(pcatest$x[,1:15])
  pca.testframe$match = test$match

  pca_test_pred <- predict(pca_log, newdata=pca.testframe[,-16], type = "response");
  pca_test_pred = ifelse(pca_test_pred > .5,1,0)
  pca_err <- mean(pca_test_pred!= test$match)
  
  #Naive Bayes
  nb_model <- naiveBayes(as.factor(match) ~. , data = train)
  nb_err <- mean(predict(nb_model, newdata = test) != test$match)
  
  #LDA
  lda_model <- lda(train[,-57], train$match)
  lda_err <- mean(predict(lda_model, test[,-57])$class != test$match)
  
  #QDA
  qda_model <- qda(train[,-57], train[,57])
  qda_err <- mean(predict(qda_model, test[,-57])$class != test$match)
  
  #KNN
  knn_model <- kknn(match~ .,
             train = train,
             test = test,
             k =  26,
             kernel = "optimal",
             scale = TRUE)
  knn_pred <- predict(knn_model)
  knn_pred = ifelse(knn_pred > .5,1,0)
  knn_err <- mean(knn_pred != test$match)
  
  cverror <- cbind(log_err, step_err, pca_err, nb_err, lda_err, qda_err, knn_err)
  CVALL <- rbind(CVALL, cverror)
}

colnames(CVALL) <- c("Logistic Regression", "Stepwise Logistic Regression", "PCA Log. Regression 15 PCs", "Naive Bayes", "LDA", "QDA", "KNN")
```

# Performance

``` r
#Train, Test, CV Error and Variance
Model <- c("Logistic Regression", "Stepwise Logistic Regression", "PCA Log. Regression with 15 PCs", "Naive Bayes", "LDA", "QDA", "KNN k = 26", "Random Forest", "Gradient Boosted")

CVerr <- apply(CVALL, 2, mean)
CVerr <- c(CVerr, "NA", "NA")
CVvar <- apply(CVALL, 2, var)
CVvar <- c(CVvar, "NA", "NA")
TrainErr <- c(log_train_err, step_train_err, pca_train_err, nb_train_err, lda_train_err, qda_train_err, knn_train_err, rf_train_error, gb_train_err)
TestErr <- c(log_test_err, step_test_err, pca_test_err, nb_test_err, lda_test_err, qda_test_err, knn_test_err, rf_test_err, gb_test_err)

all_errors_df <- data_frame(Model, TrainErr, TestErr, CVerr, CVvar)
```

    ## Warning: `data_frame()` was deprecated in tibble 1.1.0.
    ## ℹ Please use `tibble()` instead.
    ## This warning is displayed once every 8 hours.
    ## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
    ## generated.

``` r
colnames(all_errors_df)[2] ="Training Error"
colnames(all_errors_df)[3] ="Test Error"
colnames(all_errors_df)[4] ="Cross Validation Error"
colnames(all_errors_df)[5] ="Cross Validation Variance"

all_errors_df
```

    ## # A tibble: 9 × 5
    ##   Model                     `Training Error` `Test Error` Cross Validation Err…¹
    ##   <chr>                     <chr>                   <dbl> <chr>                 
    ## 1 Logistic Regression       0.1384661295135…        0.143 0.14409307875895      
    ## 2 Stepwise Logistic Regres… 0.13682482840943        0.142 0.143496420047733     
    ## 3 PCA Log. Regression with… 0.1603998806326…        0.220 0.205608591885442     
    ## 4 Naive Bayes               0.2141151894956…        0.270 0.218615751789976     
    ## 5 LDA                       0.1424947776783…        0.143 0.145584725536993     
    ## 6 QDA                       0.1445837063563…        0.183 0.178371121718377     
    ## 7 KNN k = 26                NA                      0.153 0.155340095465394     
    ## 8 Random Forest             NA                      0.136 NA                    
    ## 9 Gradient Boosted          0.1005669949268…        0.132 NA                    
    ## # ℹ abbreviated name: ¹​`Cross Validation Error`
    ## # ℹ 1 more variable: `Cross Validation Variance` <chr>

## Specificity and Sensitivity

``` r
#confusion matrix logistic regression
log_conf <- confusionMatrix(data = factor(log_test_pred), reference = factor(test$match))

#confusion matrix step-wise logistic regression
step_conf <- confusionMatrix(data = factor(step_test_pred), reference = factor(test$match))

#confusion matrix PCA Log. Regression with 15 PCs
pca_conf <- confusionMatrix(data = factor(pca_test_pred), reference = factor(test$match))

#confusion matrix Naive Bayes logistic regression
nb_test_pred <- as.integer(as.logical(nb_test_pred))
nb_conf <- confusionMatrix(data = factor(nb_test_pred), reference = factor(test$match))

#confusion matrix LDA
lda_conf <- confusionMatrix(data = factor(lda_test_pred), reference = factor(test$match))

#confusion matrix QDA logistic regression
qda_test_pred <- as.integer(as.logical(qda_test_pred))
qda_conf <- confusionMatrix(data = factor(qda_test_pred), reference = factor(test$match))

#confusion matrix gradient boosted logistic regression
gb_conf <- confusionMatrix(data = factor(gb_test_pred), reference = factor(test$match))
                           
#confusion matrix Random forest
rf_conf <- confusionMatrix(data = factor(rf.pred), reference = factor(test$match))

#KNN confusion matrix
knn_conf <- confusionMatrix(data = factor(knn_pred), reference = factor(test$match))

sensitivity <- c(log_conf$byClass[1], step_conf$byClass[1], pca_conf$byClass[1],
                 nb_conf$byClass[1], lda_conf$byClass[1], qda_conf$byClass[1], 
                 knn_conf$byClass[1], rf_conf$byClass[1], gb_conf$byClass[1])
                  
specificity <- c(log_conf$byClass[2], step_conf$byClass[2], pca_conf$byClass[2],
                 nb_conf$byClass[2], lda_conf$byClass[2], qda_conf$byClass[2], 
                 knn_conf$byClass[2], rf_conf$byClass[2], gb_conf$byClass[2])

sens_spec_df <- data_frame(Model, sensitivity, specificity)

colnames(sens_spec_df)[1] ="Model"
colnames(sens_spec_df)[2] ="Sensitivity"
colnames(sens_spec_df)[3] ="Specificity"

sens_spec_df
```

    ## # A tibble: 9 × 3
    ##   Model                           Sensitivity Specificity
    ##   <chr>                                 <dbl>       <dbl>
    ## 1 Logistic Regression                   0.967      0.313 
    ## 2 Stepwise Logistic Regression          0.969      0.313 
    ## 3 PCA Log. Regression with 15 PCs       0.941      0.112 
    ## 4 Naive Bayes                           0.789      0.228 
    ## 5 LDA                                   0.924      0.0485
    ## 6 QDA                                   0.814      0.168 
    ## 7 KNN k = 26                            0.982      0.179 
    ## 8 Random Forest                         0.945      0.0485
    ## 9 Gradient Boosted                      0.906      0.0821

## Chi-Sq test of Logistic Regression Models

``` r
#Chi-Sq test to compare Logistic Regression model full vs. stepwise reduced
anova(log_model, step_model, test = "Chisq")
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: match ~ wave + samerace + importance_same_race + importance_same_religion + 
    ##     pref_o_attractive + pref_o_sincere + pref_o_intelligence + 
    ##     pref_o_funny + pref_o_ambitious + pref_o_shared_interests + 
    ##     attractive_o + sinsere_o + intelligence_o + funny_o + ambitous_o + 
    ##     shared_interests_o + attractive_important + sincere_important + 
    ##     intellicence_important + funny_important + ambtition_important + 
    ##     shared_interests_important + attractive + sincere + intelligence + 
    ##     funny + ambition + attractive_partner + sincere_partner + 
    ##     intelligence_partner + funny_partner + ambition_partner + 
    ##     shared_interests_partner + sports + tvsports + exercise + 
    ##     dining + museums + art + hiking + gaming + clubbing + reading + 
    ##     tv + theater + movies + concerts + music + shopping + yoga + 
    ##     interests_correlate + expected_happy_with_sd_people + expected_num_matches + 
    ##     like + guess_prob_liked + met + age_diff
    ## Model 2: match ~ importance_same_race + importance_same_religion + pref_o_intelligence + 
    ##     pref_o_funny + attractive_o + funny_o + ambitous_o + shared_interests_o + 
    ##     sincere_important + intellicence_important + attractive + 
    ##     intelligence + attractive_partner + sincere_partner + funny_partner + 
    ##     ambition_partner + shared_interests_partner + tvsports + 
    ##     museums + art + clubbing + tv + movies + shopping + yoga + 
    ##     expected_num_matches + like + guess_prob_liked + age_diff
    ##   Resid. Df Resid. Dev  Df Deviance Pr(>Chi)
    ## 1      6644     4368.3                      
    ## 2      6672     4382.6 -28  -14.353   0.9845

``` r
#the full model is statistically significantly better than the stepwise-reduced 
```
