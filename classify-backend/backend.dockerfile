FROM tiangolo/uvicorn-gunicorn-machine-learning:python3.7

RUN conda install faiss-cpu=1.5.1 -c pytorch -y
RUN conda install -c anaconda certifi -y

RUN pip install celery~=4.3 passlib[bcrypt] tenacity requests emails "fastapi>=0.16.0" uvicorn gunicorn pyjwt python-multipart email_validator jinja2 psycopg2-binary alembic SQLAlchemy pydicom==1.3 fastai==1.0.57 aiofiles==0.4.0

RUN mkdir -p /root/.cache/torch/checkpoints/
RUN wget "https://download.pytorch.org/models/resnet34-333f7ec4.pth" -O "/root/.cache/torch/checkpoints/resnet34-333f7ec4.pth"

# For development, Jupyter remote kernel, Hydrogen
# Using inside the container:
# jupyter lab --ip=0.0.0.0 --allow-root --NotebookApp.custom_display_url=http://127.0.0.1:8888
ARG env=prod
RUN bash -c "if [ $env == 'dev' ] ; then pip install jupyterlab ; fi"
EXPOSE 8888

COPY ./app /app
WORKDIR /app/

ENV PYTHONPATH=/app

EXPOSE 80
