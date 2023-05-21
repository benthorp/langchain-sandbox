import axios from 'axios';
import cheerio from 'cheerio';

export async function parseSitemap(sitemapUrl: string): Promise<string[]> {
  try {
    const response = await axios.get(sitemapUrl);
    const sitemapXml = response.data;
    const $ = cheerio.load(sitemapXml, { xmlMode: true });

    const urls: string[] = [];
    $('urlset > url > loc').each((index, element) => {
      const url = $(element).text();
      urls.push(url);
    });

    return urls;
  } catch (error) {
    console.error('Error parsing sitemap:', error);
    return [];
  }
}
