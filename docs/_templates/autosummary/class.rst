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
         {%- if not item.startswith('_') or item in ['__call__', '__getitem__', '__setitem__', '__len__', '__repr__', '__str__', '__array__', '__array_ufunc__', '__array_function__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__add__', '__sub__', '__mul__', '__matmul__', '__truediv__', '__floordiv__', '__mod__', '__divmod__', '__pow__', '__lshift__', '__rshift__', '__and__', '__xor__', '__or__', '__neg__', '__pos__', '__abs__', '__invert__'] %}
         ~{{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
      {% for item in inherited_members %}
         {%- if item in ['__call__', '__getitem__', '__setitem__', '__len__', '__repr__', '__str__', '__array__', '__array_ufunc__', '__array_function__', '__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__', '__add__', '__sub__', '__mul__', '__matmul__', '__truediv__', '__floordiv__', '__mod__', '__divmod__', '__pow__', '__lshift__', '__rshift__', '__and__', '__xor__', '__or__', '__neg__', '__pos__', '__abs__', '__invert__'] %}
         ~{{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
  {% endblock %}
