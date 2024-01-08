## AI 기반 뉴스 연관 키워드 매칭 모형 개발
### Case Study를 위한 설명

1. 모델 파일 (.pth)과 키워드를 예측하고 싶은 뉴스 내용을 담은 파일 (.txt)를 준비합니다.
2. case_study.py 파일을 나머지 py 파일들과 동일한 위치에 옮겨줍니다.
3. 몇가지 args를 지정해줍니다.
   - model_dir : 모델 파일 (.pth)의 위치
   - news_dir : 예측하고 싶은 뉴스 내용을 담은 파일 (.txt.)의 위치 
   - distance : 모델에서 사용한 distance 방법 ['euclidean distance', 'cosine similarity']
   - k : 얼만큼의 키워드를 예측하고 싶은지 지정
4. args를 지정해서 case_study.py를 돌려줍니다
```ruby
python case_study.py --model_dir ./data/kobert.pth --news_dir ./data/news.txt --distance euclidean distance --k 3
```
