from pathlib import Path
p = Path(__file__).parent / 'template.html'; s = p.read_text(encoding='utf8')
if 'deep_mtg.html' not in s:
    s = s.replace('<a href="#caveats">Caveats</a></nav>',
                  '<a href="#caveats">Caveats</a><span>Deep dives:</span><a href="deep_mtg.html">Mind the Gap</a><a href="deep_tf.html">Teacher forcing</a><a href="deep_pipeline.html">Tail pipeline</a></nav>')
    s = s.replace('<h3>How Mind the Gap works</h3>', '<h3>How Mind the Gap works <a class="dd" href="deep_mtg.html">deep dive →</a></h3>')
    s = s.replace('<h3>What the tail-vote pipeline taught us (last three weeks)</h3>', '<h3>What the tail-vote pipeline taught us (last three weeks) <a class="dd" href="deep_pipeline.html">deep dive →</a></h3>')
    s = s.replace('<h2 id="own">Teacher forcing vs the model\'s own answers</h2>', '<h2 id="own">Teacher forcing vs the model\'s own answers <a class="dd" href="deep_tf.html">deep dive →</a></h2>')
    s = s.replace('<p><b>Teacher forcing.</b>', '<p><b>Teacher forcing</b> (<a href="deep_tf.html">deep dive</a>).')
    s = s.replace('.note{', 'a.dd{font-family:var(--body);font-size:14px;font-weight:400;color:var(--accent)}\nnav span{color:var(--muted)}\n.note{')
    p.write_text(s, encoding='utf8')
print('links:', s.count('deep_'))
