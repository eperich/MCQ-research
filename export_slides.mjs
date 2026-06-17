import puppeteer from 'puppeteer';
import { mkdirSync } from 'fs';
import { resolve } from 'path';

const FILE = resolve('./asee_presentation.html');
const OUT  = './slide_exports';

mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({ headless: true });
const page    = await browser.newPage();

// 1920×1080 = standard 16:9, perfect for Google Slides
await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 2 });
await page.goto(`file://${FILE}`, { waitUntil: 'networkidle0' });

// Hide UI chrome that shouldn't appear in exports
await page.evaluate(() => {
  const btn = document.getElementById('fsBtn');
  if (btn) btn.style.display = 'none';
  const prog = document.querySelector('.progress');
  if (prog) prog.style.display = 'none';
});

// Count slides
const total = await page.evaluate(() =>
  document.querySelectorAll('.slide').length
);
console.log(`Found ${total} slides`);

for (let i = 0; i < total; i++) {
  // Activate the target slide directly via JS
  await page.evaluate((idx) => {
    const slides = document.querySelectorAll('.slide');
    slides.forEach((s, j) => {
      s.classList.toggle('active', j === idx);
    });
    // Disable transitions so screenshot captures final state
    slides[idx].style.transition = 'none';
    slides[idx].style.opacity = '1';
    slides[idx].style.transform = 'none';
  }, i);

  // Brief pause for any repaints
  await new Promise(r => setTimeout(r, 120));

  const num  = String(i + 1).padStart(2, '0');
  const path = `${OUT}/slide_${num}.png`;
  await page.screenshot({ path, fullPage: false });
  console.log(`  saved slide_${num}.png`);
}

await browser.close();
console.log(`\nDone! ${total} PNGs in ./${OUT}`);
