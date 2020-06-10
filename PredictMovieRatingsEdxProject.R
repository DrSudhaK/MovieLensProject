#----------------------------------------------------------------------------------------
## Sudha Kankipati
## MovieLens Project 
## HarvardX: PH125.9x Data Science: Capstone 
## Running on  Windows 10, RAM 32GB 
#----------------------------------------------------------------------------------------
#Introduction
#----------------------------------------------------------------------------------------
#Recommendation systems use rating data from many products and users to make 
#recommendations for a specific user.  
#Will use the following code to generate datasets and develop algorithm using the edx set.
#For a final test of algorithm, We predict movie ratings in the validation set
#as if they were unknown.RMSE will be used to evaluate how close predictions are
#to the true values in the validation set.

#----------------------------------------------------------------------------------------
#Data Preparation
#----------------------------------------------------------------------------------------
# Create edx set, validation set
#----------------------------------------------------------------------------------------

# Note: this process could take a couple of minutes

if(!require(tidyverse))install.packages("tidyverse", repos ="http://cran.us.r-project.org")
if(!require(caret))install.packages("caret", repos ="http://cran.us.r-project.org")
if(!require(data.table))install.packages("data.table", repos ="http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))


movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#----------------------------------------------------------------------------------------
#Data Exploration
#----------------------------------------------------------------------------------------
# Before creating the model,Understanding the features of the rating data set.
# This step will help build a better model.

#structure of dataset
str(edx)

# Head
head(edx) %>%
  print.data.frame()

# Total unique movies and users
summary(edx)

# Number of unique movies and users in the edx dataset 
edx %>%
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))

# Ratings distribution
edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "White") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating distribution")

# Plot number of ratings per movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "green") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")

# showing train set
edx %>% as_tibble()

edx %>% summarize(
  n_users=n_distinct(userId),# unique users from train set
  n_movies=n_distinct(movieId),# unique movies from train set
  min_rating=min(rating), # the lowest rating in train set
  max_rating=max(rating) # the highest rating in train set
)

if(!require(dataMaid)) install.packages("dataMaid", repos = "http://cran.us.r-project.org")
library(dataMaid)
makeDataReport(edx, replace=TRUE)

#----------------------------------------------------------------------------------------
##  Data Cleaning
#----------------------------------------------------------------------------------------
# Removing genres and timestamp.
edx <- edx %>% select(userId, movieId, rating, title)
validation <- validation %>% select(userId, movieId, rating, title)

#----------------------------------------------------------------------------------------
##  Data Visualization
#----------------------------------------------------------------------------------------
# Movies exploration
#----------------------------------------------------------------------------------------
# Checking number of different movies present in 'edx' set
length(unique(edx$movieId))

# Distribution of movies: Movies rated more than others (histogram)
edx %>% group_by(movieId) %>%
  summarise(n=n()) %>%
  ggplot(aes(n)) +
  geom_histogram(bin = 30, color = "white") +
  scale_x_log10() + 
  ggtitle("Movies")

#----------------------------------------------------------------------------------------
# Users exploration
#----------------------------------------------------------------------------------------
# Checking number of different users present in 'edx' set
length(unique(edx$userId))

# Distribution of users
edx %>% group_by(userId) %>%
  summarise(n=n()) %>%
  arrange(n) %>%
  head()


#----------------------------------------------------------------------------------------
# Insights
#----------------------------------------------------------------------------------------
#The first thing we notice is that some movies get rated more than others.
# Here is the distribution: plot count rating by movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "White") +
  scale_x_log10() +
  ggtitle("Movies")

#Our second observation is that some users are more active than others at rating movies:
# plot count rating by user
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "green") +
  scale_x_log10() +
  ggtitle("Users")


if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(rmarkdown)) install.packages("rmarkdown", repos = "http://cran.us.r-project.org")
if(!require(markdown)) install.packages("markdown", repos = "http://cran.us.r-project.org")
library(knitr)
library(rmarkdown)
library(markdown)

#----------------------------------------------------------------------------------------
#Methods and Analysis
#----------------------------------------------------------------------------------------
# RMSE
#----------------------------------------------------------------------------------------
#Root Mean Squared Error (RMSE) is the indicator used to compare the predicted value 
#with the actual outcome. During the model development, we use the test(edx) set to 
#predict the outcome. 
#When the model is ready, then we use the 'validation' set.

#----------------------------------------------------------------------------------------
# Linear Model
#----------------------------------------------------------------------------------------

## Average movie rating model ##

## compute mean rating
mu <- mean(edx$rating) 
mu
## compute root mean standard error
naive_rmse <- RMSE(edx$rating, mu) 

# Check results
# Save prediction in tibble
rmse_results <- tibble(method = "Average movie rating model", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

## Movie effect model ##

# Simple model taking into account the movie effect b_i
# Subtract the rating minus the mean for each rating the movie received
# Plot number of movies with the computed b_i
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("White"),
                     ylab = "Number of movies", main = "Number of movies with the computed b_i")

# Test and save rmse results 
predicted_ratings <- mu +  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie effect model",  
                                     RMSE = model_1_rmse ))
# Check results
rmse_results %>% knitr::kable()


#Matrix factorization
# recosystem is a package for recommendation system
# using Matrix Factorization. High performance multi-core
# parallel computing is supported in this package.


if(!require(recosystem)) 
  install.packages("recosystem", repos = "http://cran.us.r-project.org")
set.seed(1)

# This is a randomized algorithm

# Convert the train and test sets into recosystem input format
train_data <-  with(edx, data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))
test_data  <-  with(validation,  data_memory(user_index = userId, 
                                           item_index = movieId, 
                                           rating     = rating))

# Create the model object
recommender <-  recosystem::Reco()

# Select the best tuning parameters
opts <- recommender$tune(train_data, opts = list(dim = c(10, 20, 30), 
                                       lrate = c(0.1, 0.2),
                                       costp_l2 = c(0.01, 0.1), 
                                       costq_l2 = c(0.01, 0.1),
                                       nthread  = 4, niter = 10))

# Training the algorithm  
recommender$train(train_data, opts = c(opts$min, nthread = 4, niter = 20))


# Calculate the predicted values  

#We predict unknown entries in the rating matrix on test set.
# use the `$predict()` method to compute predicted values
# return predicted values in memory
predicted_ratings <-  recommender$predict(test_data, out_memory())
head(predicted_ratings, 10)#We use the test set for the final assessment

# ceiling rating at 5
ind <- which(predicted_ratings > 5)
predicted_ratings[ind] <- 5

# floor rating at 0.50
ind <- which(predicted_ratings < 0.5)
predicted_ratings[ind] <- 0.5

# create a results table with this approach
model_2_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie and User + Matrix Fact. on test set",
                                 RMSE = model_2_rmse))
rmse_results %>% knitr::kable()
set.seed(1)

# Convert 'edx' and 'validation' sets to recosystem input format
edx_reco <-  with(edx, data_memory(user_index = userId, 
                                   item_index = movieId, 
                                   rating = rating))
validation_reco  <-  with(validation, data_memory(user_index = userId, 
                                                  item_index = movieId, 
                                                  rating = rating))

# Create the model object
recommender <-  recosystem::Reco()

# Tune the parameters
opts <-  recommender$tune(edx_reco, opts = list(dim = c(10, 20, 30), 
                                      lrate = c(0.1, 0.2),
                                      costp_l2 = c(0.01, 0.1), 
                                      costq_l2 = c(0.01, 0.1),
                                      nthread  = 4, niter = 10))

# Training the model
recommender$train(edx_reco, opts = c(opts$min, nthread = 4, niter = 20))

# create a results table with this and prior approaches
# Calculate the prediction
predicted_ratings <-  recommender$predict(validation_reco, out_memory())


model_3_rmse <- RMSE(validation$rating, predicted_ratings)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Movie and User effects and Matrix Fact. on validation set",
                                 RMSE = model_3_rmse))
rmse_results %>% knitr::kable()

#Summary
#We reach a RMSE of
model_3_rmse
#on validation set using the full edx set for training. 
#The model Movie and User effects and Matrix Factorization achieved this performance using the
#methodology presented in the course's book 
#(https://rafalab.github.io/dsbook/largedatasets.html#recommendation-systems)

####information####
print("Operating System:")
version
