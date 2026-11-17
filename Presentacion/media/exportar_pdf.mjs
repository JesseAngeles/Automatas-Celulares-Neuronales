// Exporta la presentación a PDF, una página de 1920 × 1080 por diapositiva.
//   node media/exportar_pdf.mjs [salida.pdf]
// Cada video se cambia por uno de sus cuadros (CUADRO), se abre una copia temporal
// de index.html en Firefox sin ventana (flatpak) y se imprime con WebDriver BiDi.
// Al final Ghostscript recomprime las imágenes. Las notas del presentador no se incluyen.
import { readFileSync, writeFileSync, mkdirSync, rmSync } from 'node:fs';
import { execFileSync, spawn } from 'node:child_process';
import { homedir } from 'node:os';
import path from 'node:path';

const deck = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const salida = path.resolve(process.argv[2] || path.join(deck, 'TT_2026-A114_Presentacion.pdf'));
const tmp = path.join(deck, '.pdf_tmp');
const html = path.join(deck, '.pdf_tmp.html');
const perfil = path.join(homedir(), 'Downloads', '.pdf_tmp_firefox');  // flatpak solo escribe en ~/Downloads
const puerto = 9333;

// Fracción de la duración del cuadro que se usa; los demás videos usan el último cuadro
const CUADRO = { gol_aleatoria: 0.5, flame_plot_1: 0.5, flame_plot_2: 0.5, flame_centro: 0.35, sevir_mejor: 0.5 };

function cuadro(nombre) {
  const src = path.join(deck, 'media/videos', nombre + '.webm');
  const png = path.join(tmp, nombre + '.png');
  const f = CUADRO[nombre];
  const pos = f === undefined ? ['-sseof', '-0.3'] : ['-ss', String(f * duracion(src))];
  execFileSync('ffmpeg', ['-v', 'error', '-y', ...pos, '-i', src, '-frames:v', '1', '-update', '1', png]);
  return path.relative(deck, png);
}
function duracion(src) {
  return parseFloat(execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', src]));
}

const IMPRESION = `<style>
  @page { size: 1920px 1080px; margin: 0; }
  html, body { height: auto !important; overflow: visible !important; background: var(--bg) !important; }
  #deck { position: static !important; transform: none !important; height: auto !important; }
  .slide { display: flex !important; position: relative !important; inset: auto !important;
           width: 1920px; height: 1080px; overflow: hidden; break-after: page; }
  .slide:last-of-type { break-after: auto; }
  #notes { display: none !important; }
</style>
</head>`;

// 1. Copia de index.html con cuadros en lugar de videos
rmSync(tmp, { recursive: true, force: true });
mkdirSync(tmp);
let texto = readFileSync(path.join(deck, 'index.html'), 'utf8');
texto = texto.replace(/<video src="media\/videos\/([\w-]+)\.webm"([^>]*)><\/video>/g, (_, nombre, attrs) => {
  const estilo = (attrs.match(/style="[^"]*"/) || [''])[0];
  return `<img src="${cuadro(nombre)}" alt="" ${estilo}>`;
});
texto = texto.replace('</head>', IMPRESION);
writeFileSync(html, texto);

// 2. Firefox sin ventana con WebDriver BiDi
rmSync(perfil, { recursive: true, force: true });
mkdirSync(perfil, { recursive: true });
const ff = spawn('flatpak', ['run', 'org.mozilla.firefox', '--headless', '--no-remote', '--profile', perfil,
  '--remote-debugging-port', String(puerto), 'about:blank'], { stdio: 'ignore' });
const salio = new Promise(r => ff.on('exit', r));

async function conectar() {
  for (let i = 0; i < 60; i++) {
    try {
      const ws = new WebSocket(`ws://127.0.0.1:${puerto}/session`);
      await new Promise((ok, mal) => { ws.onopen = ok; ws.onerror = mal; });
      return ws;
    } catch { await new Promise(r => setTimeout(r, 500)); }
  }
  throw new Error('Firefox no abrió el puerto ' + puerto);
}

let ws, id = 0;
const pendientes = new Map();
function pedir(method, params = {}) {
  return new Promise((ok, mal) => {
    pendientes.set(++id, { ok, mal });
    ws.send(JSON.stringify({ id, method, params }));
  });
}

try {
  ws = await conectar();
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    const p = pendientes.get(m.id);
    if (!p) return;
    pendientes.delete(m.id);
    m.type === 'error' ? p.mal(new Error(`${m.error}: ${m.message}`)) : p.ok(m.result);
  };
  await pedir('session.new', { capabilities: {} });
  const { contexts } = await pedir('browsingContext.getTree');
  const context = contexts[0].context;
  await pedir('browsingContext.navigate', { context, url: 'file://' + html, wait: 'complete' });
  const r = await pedir('script.evaluate', {
    target: { context }, awaitPromise: true,
    expression: `(async () => {
      await document.fonts.ready;
      await Promise.all([...document.images].map(i => i.decode().catch(() => {})));
      return document.querySelectorAll('.slide').length + ' ' + document.querySelectorAll('.falta').length;
    })()`,
  });
  const [n, faltan] = r.result.value.split(' ').map(Number);
  if (faltan) console.warn(`Aviso: ${faltan} archivo(s) no cargaron (recuadro "No se pudo cargar")`);
  const { data } = await pedir('browsingContext.print', {
    context, background: true, shrinkToFit: false,
    margin: { top: 0, bottom: 0, left: 0, right: 0 },
    page: { width: 50.8, height: 28.575 },  // 1920 × 1080 px a 96 ppp
  });
  // Firefox guarda las imágenes sin pérdida (15 MB); Ghostscript las pasa a JPEG cuando conviene
  const crudo = path.join(tmp, 'crudo.pdf');
  writeFileSync(crudo, Buffer.from(data, 'base64'));
  execFileSync('gs', ['-q', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.6',
    '-dPDFSETTINGS=/printer', '-sOutputFile=' + salida, crudo]);
  console.log(`${n} diapositivas -> ${salida}`);
  await pedir('browser.close').catch(() => {});
} finally {
  ws?.close();
  await Promise.race([salio, new Promise(r => setTimeout(r, 10000))]);  // el perfil se borra cuando Firefox ya cerró
  ff.kill();
  rmSync(tmp, { recursive: true, force: true });
  rmSync(html, { force: true });
  rmSync(perfil, { recursive: true, force: true });
}
