{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

  {% block attributes %}
  {% if attributes %}
      .. rubric:: Attributes

      .. autosummary::
         :toctree:
      {% for item in all_attributes %}
         {%- if not item.startswith('_') %}
         ~{{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
  {% endif %}
  {% endblock %}

  {% block methods %}
      .. rubric:: Methods

      .. autosummary::
         :toctree:

      {% for item in all_methods %}
         {%- if not item.startswith('_') or item in ['__call__', '__mul__', '__getitem__', '__len__', '__and__', '__eq__', '__or__'] %}
         ~{{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
      {% for item in inherited_members %}
         {%- if item in ['__call__', '__mul__', '__getitem__', '__len__', '__and__', '__eq__', '__or__'] %}
         ~{{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
  {% endblock %}
