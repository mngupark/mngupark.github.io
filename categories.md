---
layout: default
title: Categories
---

# Categories

Browse all posts by categories.

{% for category in site.categories offset:1 %}
  <button class="btn" id="{{ category[0] }}_btn" onclick="toggle_category('{{ category[0] }}')"><span>{{ category[0] | upcase }}</span></button>
  <ul id="{{ category[0] }}" style="display: none;">
    {% assign sorted_posts = category[1] | sort:"post-order" %}
    {% for post in sorted_posts %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}