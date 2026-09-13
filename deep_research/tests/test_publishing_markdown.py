import unittest

from deep_research.publishing.renderers.markdown import render_markdown
from deep_research.publishing.renderers.wechat import render_wechat_html
from deep_research.publishing.inline_images import extract_inline_image_references


class PublishingMarkdownRendererTests(unittest.TestCase):
    def test_basic_markdown_is_rendered(self):
        rendered = render_markdown(
            "# Title\n\n**important** and *emphasis*\n\n"
            "- first\n- second"
        )

        self.assertIn("<h1>Title</h1>", rendered)
        self.assertIn("<strong>important</strong>", rendered)
        self.assertIn("<em>emphasis</em>", rendered)
        self.assertIn("<ul>", rendered)
        self.assertIn("<li>first</li>", rendered)
        self.assertIn("<li>second</li>", rendered)

    def test_raw_html_is_escaped(self):
        rendered = render_markdown(
            '<script>alert("xss")</script>\n\n'
            '<img src="x" onerror="alert(1)">'
        )

        self.assertNotIn("<script", rendered.lower())
        self.assertNotIn("<img", rendered.lower())
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&lt;img", rendered)

    def test_dangerous_link_protocol_is_not_emitted(self):
        rendered = render_markdown(
            "[danger](javascript:alert(1))\n\n"
            "[also-dangerous](data:text/html,test)"
        )

        self.assertNotIn("javascript:", rendered.lower())
        self.assertNotIn("data:text", rendered.lower())
        self.assertNotIn("<a", rendered.lower())

    def test_external_links_have_safe_attributes(self):
        rendered = render_markdown(
            "[OpenAI](https://openai.com) and [local](/articles/example/)"
        )

        self.assertIn(
            'href="https://openai.com" target="_blank" '
            'rel="noopener noreferrer"',
            rendered,
        )
        self.assertIn('href="/articles/example/"', rendered)
        self.assertNotIn(
            'href="/articles/example/" target="_blank"',
            rendered,
        )

    def test_code_block_is_escaped(self):
        rendered = render_markdown(
            "```python\n"
            "<script>alert(1)</script>\n"
            "```"
        )

        self.assertIn("<pre><code>", rendered)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertNotIn("<script>", rendered)

    def test_invalid_input_type_is_rejected(self):
        with self.assertRaises(TypeError):
            render_markdown(None)

    def test_attachment_images_are_typed_and_can_be_resolved(self):
        markdown = 'Before\n\n![A & B](attachment://image-1)\n\nAfter'
        references = extract_inline_image_references(markdown)

        self.assertEqual(
            references[0].attachment_id,
            "image-1",
        )
        rendered = render_markdown(
            markdown,
            image_renderer=lambda attachment_id, alt: (
                f'<img src="https://cdn.example/{attachment_id}" alt="{alt}">' 
            ),
        )
        self.assertIn('src="https://cdn.example/image-1"', rendered)
        self.assertIn('alt="A & B"', rendered)
        self.assertLess(rendered.index("Before"), rendered.index("<img"))
        self.assertLess(rendered.index("<img"), rendered.index("After"))

    def test_wechat_renderer_uses_deterministic_theme_colors(self):
        rendered = render_wechat_html(
            "# Title\n\nText\n\n```python\nprint('ok')\n```",
            theme="tech",
        )

        self.assertIn("color:#0f172a", rendered)
        self.assertIn("color:#334155", rendered)
        self.assertIn("background:#f0f9ff", rendered)
        self.assertNotIn("<style", rendered)

    def test_wechat_renderer_rejects_unknown_theme(self):
        with self.assertRaises(ValueError):
            render_wechat_html("# Title", theme="unknown")


if __name__ == "__main__":
    unittest.main()
