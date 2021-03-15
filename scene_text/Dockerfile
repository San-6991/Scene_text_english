FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip sudo

RUN useradd -m admin

#RUN chown -R vision:vision /home/vision/Downloads/sunu/

#COPY --chown=sunu . /home/vision/Downloads/sunu/app

USER admin

#RUN cd /home/vision/Downloads/sunu/app/ && pip3 install -r requirements.txt

#COPY requirement.txt /home/vision/Downloads/sunu/requirement.txt
#RUN rpm -ivh /var/lib/docker/tmp/docker-builder149762043/requirement.txt

#COPY requirement.txt /var/lib/docker/tmp/docker-builder860950254/requirement.txt

#COPY requirement.txt /tmp/
#RUN pip install --requirement /tmp/requirement.txt

RUN chmod +x /var/lib/docker/tmp/requirement.txt

WORKDIR /home/vision/Downloads/sunu

EXPOSE 12000

ENTRYPOINT PYTHON3 sunu_flask.py
