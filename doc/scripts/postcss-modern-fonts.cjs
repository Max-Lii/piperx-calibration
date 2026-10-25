/** KaTeX supports WOFF2 in the same modern browsers required by this site. */
module.exports = () => ({
  postcssPlugin: 'piperx-modern-katex-fonts',
  Declaration: {
    src(declaration) {
      const rule = declaration.parent;
      if (rule.type !== 'atrule' || rule.name !== 'font-face') return;
      let katex = false;
      rule.walkDecls('font-family', (family) => { katex ||= family.value.includes('KaTeX_'); });
      if (!katex) return;
      const sources = declaration.value.split(',').filter((source) => source.includes('.woff2'));
      if (sources.length) declaration.value = sources.join(',');
    },
  },
});
module.exports.postcss = true;
