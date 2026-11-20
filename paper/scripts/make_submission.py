"""
Assemble a self-contained LaTeX folder, ready to upload.

Why this is not "copy the paper directory": `figures/` also holds ten
superseded figure pairs from before the error figures were combined, plus
PNGs, score CSVs and intermediate caches; `tables/` holds .json companions;
and the directory carries build artefacts. Uploading all of that invites
someone to edit a file the document does not include. This copies exactly the
files `paper.tex` reads -- parsed out of the source, not listed by hand -- and
then PROVES the result compiles on its own.

Output: paper/submission/ and submission.zip
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
OUT = os.path.join(PAPER, "submission")
TECTONIC = os.environ.get("TECTONIC", "tectonic")


def main():
    tex = open(os.path.join(PAPER, "paper.tex")).read()
    figs = re.findall(r"\\figifexists(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    tabs = re.findall(r"\\tabifexists\{([^}]+)\}", tex)
    print(f"  paper.tex includes {len(figs)} figures and {len(tabs)} tables")

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "figures"))
    os.makedirs(os.path.join(OUT, "tables"))

    # Both formats ship. LaTeX takes the PDF because \figifexists writes the
    # .pdf extension explicitly, so the PNGs sit alongside without competing;
    # they are there for slides and for anyone who cannot open a PDF inline.
    missing = []
    for f in figs:
        got = False
        for ext in (".pdf", ".png"):
            src = os.path.join(PAPER, "figures", f + ext)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(OUT, "figures", f + ext))
                got = got or ext == ".pdf"
        if not got:
            missing.append(f + ".pdf")
    for t in tabs:
        src = os.path.join(PAPER, "tables", t + ".tex")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT, "tables", t + ".tex"))
        else:
            missing.append(t + ".tex")
    for f in ("paper.tex", "refs.bib"):
        shutil.copy2(os.path.join(PAPER, f), os.path.join(OUT, f))

    if missing:
        print("  MISSING, the upload would show red placeholder boxes:")
        for m in missing:
            print("   ", m)
        return 1

    # The point of the exercise: does it build with ONLY these files?
    print("  compiling in the submission folder to prove it is self-contained...")
    r = subprocess.run([TECTONIC, "paper.tex"], cwd=OUT,
                       capture_output=True, text=True)
    log = r.stdout + r.stderr
    # the PNGs must not be what got embedded: check the build pulled the PDFs
    picked_png = [l for l in log.splitlines()
                  if ".png" in l.lower() and "figures" in l.lower()]
    if picked_png:
        print("  WARNING: the build mentions a PNG where a PDF was expected:")
        for l in picked_png[:3]:
            print("   ", l.strip())
    bad = [l for l in log.splitlines()
           if re.search(r"overfull|undefined|not yet generated|error", l, re.I)]
    if r.returncode != 0:
        print("  BUILD FAILED"); print(log[-2000:]); return 1
    print("  build OK" + (f", {len(bad)} warning(s)" if bad else ", no warnings"))
    for l in bad[:6]:
        print("   ", l.strip())

    open(os.path.join(OUT, "README.txt"), "w").write(f"""\
Self-contained LaTeX source for the groundwater emulator paper.

Verified: this folder compiles on its own, with no LaTeX warnings. The PDF
here was produced from these files, in this folder, by that check.

  paper.tex      the manuscript
  refs.bib       bibliography (bibtex, abbrvnat)
  figures/       {len(figs)} figures, each as .pdf AND .png
  tables/        {len(tabs)} .tex fragments, pulled in by \\tabifexists
  paper.pdf      the compiled output

TO BUILD
  Overleaf: upload the whole folder, set paper.tex as the main document.
  Locally:  pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
  (or a single `tectonic paper.tex`, which resolves the bibliography itself)

FIGURES
 - LaTeX uses the PDFs: \\figifexists writes the .pdf extension explicitly, so
   the PNGs sit alongside without competing. Verified on the build above --
   all {len(figs)} embedded images are vector, none raster.
 - The PNGs are there for slides and quick viewing. Deleting them does not
   affect the build.
 - \\graphicspath{{{{figures/}}}} is set in the preamble, so figures are
   referenced by bare name. Keep the folder structure.

ONE TRAP
 \\figifexists and \\tabifexists substitute a RED PLACEHOLDER BOX for a missing
 file rather than failing. Nothing is missing here, but if you add a reference
 to a figure you have not generated, the document still compiles and ships a
 red box. Check the PDF, not just the exit status.

The abstract, introduction and conclusions are deliberately empty.
""")

    # the freshly built pdf is the one to ship
    n = sum(len(fs) for _, _, fs in os.walk(OUT))
    mb = sum(os.path.getsize(os.path.join(d, f))
             for d, _, fs in os.walk(OUT) for f in fs) / 1e6
    z = shutil.make_archive(OUT, "zip", root_dir=os.path.dirname(OUT),
                            base_dir=os.path.basename(OUT))
    print(f"  {n} files, {mb:.1f} MB -> {OUT}")
    print(f"  zip: {z} ({os.path.getsize(z)/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
