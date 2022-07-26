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
{% for posts in site.posts %}
  {% for category in posts.categories %}
  <li>
    <h3>{{ category[0] | uppercase }}</h3>    
  </li>
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
  {% endfor %}
{% endfor %}