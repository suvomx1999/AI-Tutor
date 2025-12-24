install:
	pip install -r ai_tutor_rl/requirements.txt

train:
	python ai_tutor_rl/train.py

evaluate:
	python ai_tutor_rl/evaluate.py

dashboard:
	streamlit run ai_tutor_rl/app.py

demo:
	python ai_tutor_rl/demo_topic_completion.py
api:
	python ai_tutor_rl/api.py

clean:
	rm -rf models plots
