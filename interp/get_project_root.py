#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import PurePath

def get_project_root() -> PurePath:
    return PurePath(__file__).parent
