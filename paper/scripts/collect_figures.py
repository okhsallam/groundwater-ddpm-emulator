"""
Gather every figure the paper includes into one folder, ready to download.

The figures live in paper/figures/ beside a good deal of intermediate output --
score CSVs, caches, superseded panels -- so "copy the figures directory" hands
someone the wrong set. This walks paper.tex instead and takes exactly the
files it includes, in the order it includes them, which is the order LaTeX
numbers them. Each file is therefore named for the figure number the reader
sees in the paper, so a slide can be matched to a caption without guessing.

Both formats are copied: PNG for slides, PDF for anything that will be
enlarged. A MANIFEST.csv records the mapping and the caption's first sentence.

Writes paper/figures_for_download/ and a .zip beside it.
"""
import csv
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
SRC = os.path.join(PAPER, "figures")
OUT = os.path.join(PAPER, "figures_for_download")


def first_sentence(tex, pos):
    """The caption that follows the include at `pos`, trimmed to one line."""
    m = re.search(r"\\caption\{", tex[pos:pos + 4000])
    if not m:
        return ""
    i = pos + m.end()
    depth, out = 1, []
    for ch in tex[i:i + 4000]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
    cap = "".join(out)
    cap = re.sub(r"\\(SI|num)\{([^}]*)\}(\{[^}]*\})?", r"\2", cap)
    cap = re.sub(r"\\ref\{[^}]*\}", "", cap)
    cap = re.sub(r"\\[a-zA-Z]+", "", cap)
    cap = re.sub(r"[{}$~^\\]", "", cap)
    cap = " ".join(cap.split())
    return cap.split(". ")[0][:160]


def main():
    tex = open(os.path.join(PAPER, "paper.tex")).read()
    hits = [(m.start(), m.group(1))
            for m in re.finditer(r"\\figifexists(?:\[[^\]]*\])?\{([^}]+)\}", tex)]
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    rows, missing = [], []
    for n, (pos, name) in enumerate(hits, 1):
        cap = first_sentence(tex, pos)
        got = []
        for ext in ("png", "pdf"):
            src = os.path.join(SRC, f"{name}.{ext}")
            if not os.path.exists(src):
                continue
            dst = os.path.join(OUT, f"fig{n:02d}_{name}.{ext}")
            shutil.copy2(src, dst)
            got.append(ext)
        if not got:
            missing.append((n, name))
        rows.append(dict(figure=n, file_stem=f"fig{n:02d}_{name}",
                         source=name, formats="+".join(got) or "MISSING",
                         caption=cap))

    with open(os.path.join(OUT, "MANIFEST.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "file_stem", "source",
                                          "formats", "caption"])
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(f"  Figure {r['figure']:>2}  {r['formats']:<7} {r['file_stem']}")
    print(f"\n  {len(rows)} figures -> {OUT}")
    if missing:
        print("  MISSING (referenced by paper.tex, not on disk):")
        for n, name in missing:
            print(f"    Figure {n}: {name}")
    else:
        print("  every figure the paper includes was found")

    z = shutil.make_archive(OUT, "zip", root_dir=os.path.dirname(OUT),
                            base_dir=os.path.basename(OUT))
    print(f"  zip   : {z}  ({os.path.getsize(z)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
