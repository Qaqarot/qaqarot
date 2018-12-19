# How to make translation

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