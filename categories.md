---
layout: default
title: Categories
---

# Categories

Browse all posts by categories.

<!-- {% for category in site.categories %}
  <h3>{{ category[0] | uppercase }}</h3>
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %} -->
{% assign pagecat = page.categories | join ' ' | append: ' '%}
{% assign pagecatlen = page.categories.size %}
  <h1>{{ pagecat }}</h1>
  <ul>
    {% for post in site.posts %}
    {% assign postcat = '' %}
    {% for thispostcat in post.categories limit: pagecatlen %}
      {% assign postcat = postcat | append: thispostcat %}
      {% assign postcat = postcat | append: ' ' %}
    {% endfor %}

    {% if (postcat == pagecat) %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endif %}
    {% endfor %}
  </ul>