![image](https://user-images.githubusercontent.com/117013670/202137983-10702d78-37e9-4d7b-a3b0-dac7351b5da3.png)
___________________________________________________________________________________________________________________
![Sentiment Recommendation](https://miro.medium.com/max/1100/1*PGB0w1JZslqA-hM0xGrmJw.gif)

## INTRODUCTION:
 This project aims to recommend movies based on the sentiment behind the users text, so if the user entered i am sad, the system will understand that the user is sad, and will recommend movies based on popularity based on the users feeling like comedy movies.


## Objective:
We analyze the user's emotions through what he writes and recommend movies and series related to his feelings. The ranking of programs will be based on the popularity of movies or series.



## Dataset:

#### STC IP TV:
| Features  | Meaning |
| ------------- | ------------- |
| date_  | The date that the session start  |
| user_id_maped  | The ID of the user  |
| program_name  | The name of the movie or series and the name of the season  |
| duration_seconds  | Session start time in seconds |
| program_class  | Is the program movie or series  |
| season  | The number of season contained in each program  |
| episode  | The number of episode contained in each program  |
| program_genre  | The genre of a movie or series |
| hd  | Is the progrm provide HD or no  |
| original_name  | The name of the movie or series  |

#### EMOTIONAL TONE:
| Features  | Meaning |
| ------------- | ------------- |
| ID  | The ID of the tweets  |
| TWEET  | The user tweets  |
| LABEL  | The sentiment of the tweet |

## Approaches:
- Natural Language Processing (NLP).
- Popularity based recommendation Model.

## Result:
Based on the training model we have used, The results were: 77% macro average, 76% weighted average in precision and 76% in both in recall.

<p align="center">
<img  alt="image" src="https://user-images.githubusercontent.com/117013670/202867294-4274148a-def3-427a-94d2-1d572efe562f.png" width="900" height="150">

  
<p align="center">
  
  
  
<img src="https://user-images.githubusercontent.com/117013670/202384271-b6d76184-657a-49ea-b8d0-eb23356c025f.png" width="400" height="100">

