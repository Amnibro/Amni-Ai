require "json"

module Garden
  class Plot
    attr_reader :name

    def initialize(name)
      @name = name
    end

    def to_s
      "plot #{@name}"
    end
  end
end
