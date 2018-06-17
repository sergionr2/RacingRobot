#!/bin/bash
pytest --cov-config .coveragerc --cov-report html --cov-report term --cov=. tests/
