# How to make translation

Install anaconda.
Create environment for building blueqat documentation.

```bash
$ conda create --name blueqat-doc python=3.10
$ conda activate blueqat-doc
$ conda install sphinx
```

Ensure that you are in 'doc' dir.

Extract document’s translatable messages into pot files.

```bash
$ make gettext
```

Update your locale dir for Japanese translation.

```bash
$ sphinx-intl update -p build/gettext -l ja
```

Make translated document for checking.

```bash
$ make -e SPHINXOPTS="-D language='ja'" html
```