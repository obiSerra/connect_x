FROM gcr.io/kaggle-gpu-images/python


EXPOSE 8888

RUN mkdir /workindir
WORKDIR /workindir
COPY ./workindir /workindir

RUN pip install -r requirements.txt

RUN pip install -e .

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]